import runpod
import torch
import logging
import os
import tempfile
import base64
from datetime import datetime
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
pipeline = None
device = None

def get_storage_path():
    """Detect and return the network volume mount path"""
    possible_paths = [
        "/runpod-volume",
        "/workspace", 
        "/storage",
        "/mnt/storage",
        "/tmp"  # fallback
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.W_OK):
            logger.info(f"📁 Using storage path: {path}")
            return path
    
    # Create temp directory as fallback
    os.makedirs("/tmp/videos", exist_ok=True)
    logger.warning("⚠️ Using fallback storage: /tmp/videos")
    return "/tmp/videos"

def initialize_model():
    """Initialize CogVideoX model"""
    global pipeline, device
    
    try:
        logger.info("🚀 Starting CogVideoX model initialization...")
        
        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"📱 Using device: {device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🔥 GPU: {gpu_name}")
            logger.info(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        
        # Load CogVideoX model
        model_name = os.getenv("MODEL_NAME", "THUDM/CogVideoX-5b")
        logger.info(f"📦 Loading model: {model_name}")
        
        # Initialize pipeline with optimizations
        pipeline = CogVideoXPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        
        if device == "cuda":
            pipeline = pipeline.to(device)
            # Enable memory efficient optimizations
            try:
                pipeline.enable_model_cpu_offload()
                logger.info("✅ CPU offloading enabled")
            except Exception as e:
                logger.warning(f"⚠️ CPU offloading failed: {e}")
            
            # Try VAE optimizations - handle different method names
            try:
                if hasattr(pipeline.vae, 'enable_slicing'):
                    pipeline.vae.enable_slicing()
                    logger.info("✅ VAE slicing enabled")
                elif hasattr(pipeline, 'enable_vae_slicing'):
                    pipeline.enable_vae_slicing()
                    logger.info("✅ VAE slicing enabled (legacy method)")
                else:
                    logger.info("ℹ️ VAE slicing not available")
            except Exception as e:
                logger.warning(f"⚠️ VAE slicing failed: {e}")
            
            # Try VAE tiling for memory efficiency
            try:
                if hasattr(pipeline.vae, 'enable_tiling'):
                    pipeline.vae.enable_tiling()
                    logger.info("✅ VAE tiling enabled")
            except Exception as e:
                logger.warning(f"⚠️ VAE tiling failed: {e}")
            
        logger.info("✅ CogVideoX model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {str(e)}")
        return False

def generate_video(prompt, num_inference_steps=50, guidance_scale=6.0, num_frames=40):
    """Generate video using CogVideoX and save to persistent storage"""
    global pipeline, device
    
    try:
        logger.info(f"🎬 Generating video for prompt: '{prompt}'")
        logger.info(f"⚙️ Steps: {num_inference_steps}, Guidance: {guidance_scale}, Frames: {num_frames}")
        
        if pipeline is None:
            raise Exception("Model not initialized")
        
        # Get storage path
        storage_path = get_storage_path()
        
        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 GPU cache cleared")
        
        # Generate video
        logger.info("🔄 Starting video generation...")
        
        # Use proper autocast context
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                video_frames = pipeline(
                    prompt=prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_frames=num_frames,
                    generator=torch.Generator(device=device).manual_seed(42)
                ).frames[0]
        else:
            video_frames = pipeline(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_frames=num_frames,
                generator=torch.Generator().manual_seed(42)
            ).frames[0]
        
        logger.info(f"✅ Generated {len(video_frames)} frames")
        
        # Create video filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_filename = f"cogvideo_{timestamp}_{prompt[:20].replace(' ', '_')}.mp4"
        video_path = os.path.join(storage_path, video_filename)
        
        # Export video to persistent storage
        logger.info(f"📹 Exporting video to: {video_path}")
        export_to_video(video_frames, video_path, fps=8)
        
        # Verify file exists and get size
        if not os.path.exists(video_path):
            raise Exception(f"Video file not created at {video_path}")
        
        file_size = os.path.getsize(video_path)
        logger.info(f"📁 Video file created, size: {file_size / (1024*1024):.2f} MB")
        
        # Clear GPU cache after generation
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info("🎉 Video generation completed successfully!")
        
        return {
            "success": True,
            "video_path": video_path,
            "video_filename": video_filename,
            "video_size_mb": file_size / (1024 * 1024),
            "num_frames": len(video_frames),
            "duration_seconds": len(video_frames) / 8,  # 8 FPS
            "file_size_bytes": file_size,
            "storage_location": storage_path
        }
        
    except Exception as e:
        logger.error(f"❌ Video generation failed: {str(e)}")
        # Clear GPU cache on error
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        return {
            "success": False,
            "error": str(e)
        }

def handler(event):
    """GENERATOR HANDLER with Network Volume Storage"""
    try:
        logger.info(f"📨 Received event: {event}")
        
        # Extract input from RunPod format
        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "")
        
        # Validation
        if not prompt:
            logger.error("❌ No prompt provided")
            yield {"error": "Prompt is required"}
            return
        
        if len(prompt) > 200:
            logger.error(f"❌ Prompt too long: {len(prompt)} characters")
            yield {"error": "Prompt too long (max 200 characters)"}
            return
        
        # Optional parameters with safer defaults
        num_inference_steps = min(input_data.get("num_inference_steps", 50), 50)  # Cap at 50
        guidance_scale = input_data.get("guidance_scale", 6.0)
        num_frames = min(input_data.get("num_frames", 40), 49)  # Cap at 49 for memory
        
        logger.info(f"🎯 Processing video generation for: '{prompt}'")
        logger.info(f"🔧 Parameters - Steps: {num_inference_steps}, Guidance: {guidance_scale}, Frames: {num_frames}")
        
        # Yield progress update - initialization
        yield {
            "status": "initializing",
            "message": "Starting video generation...",
            "prompt": prompt
        }
        
        # Check if model is initialized
        if pipeline is None:
            logger.info("🔄 Model not initialized, initializing now...")
            yield {
                "status": "loading_model", 
                "message": "Loading CogVideoX model..."
            }
            
            success = initialize_model()
            if not success:
                logger.error("❌ Model initialization failed")
                yield {
                    "error": "Failed to initialize CogVideoX model",
                    "status": "failed"
                }
                return
        
        # Yield progress update - generation starting
        yield {
            "status": "generating",
            "message": "Generating video frames...",
            "progress": "0%"
        }
        
        # Generate video
        start_time = datetime.now()
        logger.info(f"⏰ Starting generation at: {start_time}")
        
        result = generate_video(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.info(f"⏰ Generation completed in: {processing_time:.1f}s")
        
        if result["success"]:
            # Yield progress update - saving
            yield {
                "status": "saving",
                "message": "Saving video to storage...",
                "progress": "90%"
            }
            
            # FINAL YIELD - File information with S3 access path
            logger.info(f"✅ Successfully generated video in {processing_time:.1f}s")
            logger.info(f"📊 Video size: {result['video_size_mb']:.2f} MB")
            logger.info(f"📁 Video saved at: {result['video_path']}")
            
            # Construct S3 access information
            s3_endpoint = "https://s3api-eu-ro-1.runpod.io/"
            volume_name = "pavidwplk8"  # This is the actual S3 bucket name
            s3_object_key = result["video_filename"]
            
            yield {
                "video_filename": result["video_filename"],
                "video_path": result["video_path"],
                "prompt": prompt,
                "status": "completed",
                "success": True,
                "s3_access": {
                    "endpoint": s3_endpoint,
                    "bucket": volume_name,
                    "object_key": s3_object_key,
                    "s3_url": f"s3://{volume_name}/{s3_object_key}",
                    "download_instruction": "Use AWS CLI with RunPod S3 credentials to download"
                },
                "metadata": {
                    "model": "CogVideoX-5b",
                    "num_frames": result["num_frames"],
                    "duration_seconds": result["duration_seconds"],
                    "video_size_mb": round(result["video_size_mb"], 2),
                    "file_size_bytes": result["file_size_bytes"],
                    "processing_time_seconds": round(processing_time, 1),
                    "inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "gpu_available": torch.cuda.is_available(),
                    "device": str(device),
                    "storage_location": result["storage_location"],
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        else:
            # Generation failed
            logger.error(f"❌ Generation failed: {result['error']}")
            yield {
                "error": result["error"],
                "status": "failed",
                "success": False,
                "prompt": prompt,
                "processing_time_seconds": round(processing_time, 1),
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(f"❌ {error_msg}")
        yield {
            "error": error_msg,
            "status": "failed",
            "success": False,
            "timestamp": datetime.now().isoformat()
        }

def startup():
    """Startup function - initialize model and check storage"""
    logger.info("🚀 Starting CogVideoX service with Network Volume storage...")
    logger.info(f"🐍 Python version: {os.sys.version}")
    logger.info(f"🔥 PyTorch version: {torch.__version__}")
    logger.info(f"💻 CUDA available: {torch.cuda.is_available()}")
    
    # Check storage availability
    storage_path = get_storage_path()
    logger.info(f"📁 Storage initialized: {storage_path}")
    
    # Create videos directory if it doesn't exist
    videos_dir = os.path.join(storage_path, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    logger.info(f"📂 Videos directory: {videos_dir}")
    
    if torch.cuda.is_available():
        logger.info(f"🎮 CUDA version: {torch.version.cuda}")
        logger.info(f"🔧 GPU device: {torch.cuda.get_device_name(0)}")
        
        # Log GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"💾 Total GPU memory: {total_memory:.1f} GB")
        
        # Check available memory
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        logger.info(f"🆓 Available GPU memory: {free_memory / 1024**3:.1f} GB")
    
    # Auto-initialize model if environment variable is set
    auto_init = os.getenv("AUTO_INIT_MODEL", "false").lower() == "true"
    if auto_init:
        logger.info("🔄 AUTO_INIT_MODEL enabled, initializing model...")
        initialize_model()
    else:
        logger.info("⏳ Model will be initialized on first request")
    
    logger.info("✅ Service startup completed!")

# Run startup
startup()

# Start the RunPod serverless worker with GENERATOR HANDLER
logger.info("🌐 Starting RunPod serverless worker with Network Volume storage...")
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True  # CRITICAL: Enables streaming for /run endpoint
})