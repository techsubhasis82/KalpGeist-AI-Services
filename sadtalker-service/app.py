# Hybrid SadTalker RunPod Application
# Your API + Official SadTalker inference.py = Best of Both Worlds

from official_sadtalker_downloader import download_official_sadtalker_models, verify_sadtalker_models
from hybrid_sadtalker_processor import HybridSadTalkerProcessor  # New hybrid processor
import runpod
import torch
import logging
import os
import tempfile
import base64
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
hybrid_sadtalker = None
device = None

def get_storage_path():
    """Get network volume mount path"""
    if "RUNPOD_VOLUME_PATH" in os.environ:
        env_path = os.environ["RUNPOD_VOLUME_PATH"]
        if os.path.exists(env_path) and os.access(env_path, os.W_OK):
            logger.info(f"📁 Using environment volume: {env_path}")
            return env_path
    
    possible_paths = ["/runpod-volume", "/workspace", "/storage", "/tmp"]
    
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.W_OK):
            logger.info(f"📁 Using storage: {path}")
            return path
    
    os.makedirs("/tmp/hybrid_sadtalker", exist_ok=True)
    logger.warning("⚠️ Using fallback: /tmp/hybrid_sadtalker")
    return "/tmp/hybrid_sadtalker"

def get_model_storage_path():
    """Get model storage path"""
    base_path = get_storage_path()
    model_path = os.path.join(base_path, "official_sadtalker_models")
    os.makedirs(model_path, exist_ok=True)
    
    # Also create checkpoints subdirectory
    checkpoints_path = os.path.join(model_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    
    logger.info(f"📂 Model storage: {model_path}")
    return model_path

def initialize_hybrid_sadtalker():
    """Initialize Hybrid SadTalker (API Wrapper + Official inference.py)"""
    global hybrid_sadtalker, device
    
    try:
        logger.info("🚀 Initializing Hybrid SadTalker (API + Official)...")
        
        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"📱 Device: {device}")
        
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"⚠️ GPU info failed: {e}")
        
        # Get model storage path
        model_path = get_model_storage_path()
        
        # Check and download official models
        logger.info("🔍 Checking Official SadTalker Models...")
        if not verify_sadtalker_models(model_path):
            logger.info("📥 Downloading Official SadTalker Models...")
            
            download_success = download_official_sadtalker_models(model_path)
            if not download_success:
                logger.error("❌ Failed to download official models")
                return False
                
            # Verify after download
            if not verify_sadtalker_models(model_path):
                logger.error("❌ Model verification failed after download")
                return False
        else:
            logger.info("✅ Official SadTalker Models Verified")
        
        # Initialize Hybrid SadTalker Processor
        logger.info("📦 Loading Hybrid SadTalker Processor...")
        processor = HybridSadTalkerProcessor(device, model_path)
        
        # Get model status
        model_status = processor.get_model_status()
        logger.info(f"📊 Hybrid Status: {model_status}")
        
        # Update global model variable
        hybrid_sadtalker = {
            "initialized": True,
            "device": device,
            "ready": True,
            "model_path": model_path,
            "processor": processor,
            "version": "Hybrid_API_Official_SadTalker_v1.0",
            "architecture": "Hybrid_API_Wrapper_Official_inference.py",
            "quality_level": "Official_Maximum_Quality_Guaranteed",
            "model_status": model_status
        }
        
        logger.info("🎉 Hybrid SadTalker Loaded Successfully!")
        logger.info(f"🎭 Architecture: API Wrapper + Official inference.py")
        logger.info(f"🔧 Quality: Official SadTalker Maximum (512px + all features)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Hybrid SadTalker: {str(e)}")
        import traceback
        logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
        return False

def validate_inputs(image_data, audio_data):
    """Validate input data"""
    try:
        if not image_data:
            return False, "Image data is required"
        
        if not audio_data:
            return False, "Audio data is required"
        
        logger.info("✅ Input validation passed")
        return True, "Valid inputs"
        
    except Exception as e:
        logger.error(f"❌ Input validation failed: {str(e)}")
        return False, str(e)

def generate_hybrid_talking_head(image_base64, audio_base64, settings=None):
    """Generate talking head using Hybrid approach (API + Official)"""
    global hybrid_sadtalker, device
    
    try:
        logger.info("🎭 Starting Hybrid SadTalker Generation...")
        
        # Default settings for maximum quality
        default_settings = {
            'still': True,           # Official still mode
            'enhancer': 'gfpgan',    # Official face enhancement
            'preprocess': 'full',    # Official full preprocessing
            'size': 512,             # Official 512px
            'expression_scale': 1.0,  # Natural expressions
            'pose_style': 0          # Default pose
        }
        
        # Apply custom settings
        if settings:
            logger.info(f"🔧 Custom settings: {settings}")
            default_settings.update(settings)
        
        logger.info(f"⚙️ Final settings: {default_settings}")
        
        if hybrid_sadtalker is None:
            raise Exception("Hybrid SadTalker not initialized")
        
        # Get storage path
        storage_path = get_storage_path()
        
        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Decode inputs
        logger.info("🔄 Processing inputs...")
        try:
            image_base64_clean = image_base64.strip().replace('\n', '').replace('\r', '')
            audio_base64_clean = audio_base64.strip().replace('\n', '').replace('\r', '')
            
            image_data = base64.b64decode(image_base64_clean)
            audio_data = base64.b64decode(audio_base64_clean)
            
            logger.info(f"📷 Image decoded: {len(image_data)} bytes")
            logger.info(f"🎵 Audio decoded: {len(audio_data)} bytes")
            
            if len(image_data) < 1000:
                raise Exception(f"Image data too small: {len(image_data)} bytes")
            if len(audio_data) < 1000:
                raise Exception(f"Audio data too small: {len(audio_data)} bytes")
                
        except Exception as e:
            raise Exception(f"Failed to decode inputs: {str(e)}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save input files
        input_image_path = os.path.join(storage_path, f"input_image_{timestamp}.jpg")
        input_audio_path = os.path.join(storage_path, f"input_audio_{timestamp}.wav")
        output_video_path = os.path.join(storage_path, f"hybrid_result_{timestamp}.mp4")
        
        try:
            # Save and validate image
            with open(input_image_path, 'wb') as f:
                f.write(image_data)
            
            test_image = cv2.imread(input_image_path)
            if test_image is None:
                raise Exception(f"Invalid image file")
            logger.info(f"✅ Image saved and validated: {test_image.shape}")
            
            # Save and validate audio
            with open(input_audio_path, 'wb') as f:
                f.write(audio_data)
            
            if not os.path.exists(input_audio_path) or os.path.getsize(input_audio_path) < 1000:
                raise Exception(f"Invalid audio file")
            logger.info(f"✅ Audio saved and validated: {os.path.getsize(input_audio_path)} bytes")
            
        except Exception as e:
            raise Exception(f"Failed to save input files: {str(e)}")
        
        # Get Hybrid SadTalker processor
        processor = hybrid_sadtalker["processor"]
        
        # Process using Hybrid approach (API wrapper + Official inference.py)
        logger.info("🎬 Processing with Hybrid Architecture...")
        logger.info("📋 Pipeline: API Wrapper → Official inference.py → Maximum Quality")
        
        result = processor.process_talking_head(
            input_image_path, 
            input_audio_path, 
            output_video_path,
            settings=default_settings
        )
        
        if not result['success']:
            raise Exception(f"Hybrid processing failed: {result.get('error', 'Unknown error')}")
        
        # Verify output
        if not os.path.exists(output_video_path):
            raise Exception("Output video not created")
        
        file_size = os.path.getsize(output_video_path)
        
        logger.info("🎉 Hybrid Generation Completed Successfully!")
        logger.info(f"📊 Output: {output_video_path} ({file_size/(1024*1024):.1f}MB)")
        
        return {
            "success": True,
            "video_path": output_video_path,
            "video_filename": os.path.basename(output_video_path),
            "video_size_mb": file_size / (1024 * 1024),
            "file_size_bytes": file_size,
            "processing_time": result.get('processing_time', 0),
            "settings_used": result.get('settings_used', default_settings),
            "architecture": result.get('architecture', 'Hybrid_API_Official_SadTalker'),
            "quality_level": result.get('quality_level', 'Official_Maximum_Quality'),
            "lip_sync_quality": "Official_Professional_Grade",
            "model_version": hybrid_sadtalker.get("version"),
        }
        
    except Exception as e:
        logger.error(f"❌ Hybrid generation failed: {str(e)}")
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return {
            "success": False,
            "error": str(e)
        }

def handler(event):
    """Main handler for Hybrid SadTalker requests"""
    try:
        logger.info(f"📨 Received Hybrid SadTalker event")
        
        # Extract input
        input_data = event.get("input", {})
        image_base64 = input_data.get("image")
        audio_base64 = input_data.get("audio")
        settings = input_data.get("settings", {})
        
        # Validation
        is_valid, validation_msg = validate_inputs(image_base64, audio_base64)
        if not is_valid:
            logger.error(f"❌ Validation failed: {validation_msg}")
            yield {"error": validation_msg}
            return
        
        # Yield progress - initialization
        yield {
            "status": "initializing",
            "message": "Starting Hybrid SadTalker (API + Official)...",
            "progress": "10%",
            "architecture": "Hybrid_API_Official_SadTalker"
        }
        
        # Check if model is initialized
        if hybrid_sadtalker is None:
            logger.info("🔄 Initializing Hybrid SadTalker...")
            yield {
                "status": "loading_model", 
                "message": "Loading Hybrid SadTalker (API Wrapper + Official inference.py)...",
                "progress": "25%"
            }
            
            success = initialize_hybrid_sadtalker()
            if not success:
                logger.error("❌ Hybrid model initialization failed")
                yield {
                    "error": "Failed to initialize Hybrid SadTalker",
                    "status": "failed"
                }
                return
        
        # Yield progress - processing
        yield {
            "status": "processing",
            "message": "Processing with Official inference.py (Maximum Quality Guaranteed)...",
            "progress": "50%"
        }
        
        # Generate talking head
        start_time = datetime.now()
        
        result = generate_hybrid_talking_head(
            image_base64=image_base64,
            audio_base64=audio_base64,
            settings=settings
        )
        
        end_time = datetime.now()
        total_processing_time = (end_time - start_time).total_seconds()
        
        if result["success"]:
            # Yield progress - saving
            yield {
                "status": "saving",
                "message": "Saving Official Quality Video...",
                "progress": "90%"
            }
            
            # S3 access info
            s3_endpoint = "https://s3api-eu-ro-1.runpod.io/"
            volume_name = "pavidwplk8"
            s3_object_key = result["video_filename"]
            
            # FINAL SUCCESS YIELD
            yield {
                "video_filename": result["video_filename"],
                "video_path": result["video_path"],
                "status": "completed",
                "success": True,
                "progress": "100%",
                "s3_access": {
                    "endpoint": s3_endpoint,
                    "bucket": volume_name,
                    "object_key": s3_object_key,
                    "s3_url": f"s3://{volume_name}/{s3_object_key}"
                },
                "metadata": {
                    "model": "Hybrid_SadTalker_API_Official",
                    "architecture": result["architecture"], 
                    "quality_level": result["quality_level"],
                    "lip_sync_quality": result["lip_sync_quality"],
                    "model_version": result["model_version"],
                    "settings_used": result["settings_used"],
                    "video_size_mb": round(result["video_size_mb"], 2),
                    "processing_time_seconds": round(total_processing_time, 1),
                    "official_processing_time": round(result.get("processing_time", 0), 1),
                    "gpu_available": torch.cuda.is_available(),
                    "device": str(device),
                    "timestamp": datetime.now().isoformat(),
                    "hybrid_approach": "API_Wrapper_Plus_Official_inference.py"
                }
            }
            
        else:
            # Generation failed
            yield {
                "error": result["error"],
                "status": "failed",
                "success": False,
                "processing_time_seconds": round(total_processing_time, 1),
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
    """Startup function"""
    logger.info("🚀 Starting Hybrid SadTalker Service")
    logger.info(f"🎭 Architecture: API Wrapper + Official inference.py")
    logger.info(f"📊 Quality: Official Maximum (Guaranteed)")
    logger.info(f"🔧 Benefits: Your API + Official SadTalker Reliability")
    
    # Check storage
    storage_path = get_storage_path()
    model_path = get_model_storage_path()
    
    # GPU info
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🔥 GPU: {gpu_name} ({total_memory:.1f} GB)")
        except:
            logger.info("🔥 CUDA available")
    
    # Auto-initialize if requested
    auto_init = os.getenv("AUTO_INIT_MODEL", "false").lower() == "true"
    if auto_init:
        logger.info("🔄 AUTO_INIT_MODEL enabled, initializing Hybrid SadTalker...")
        success = initialize_hybrid_sadtalker()
        if success:
            logger.info("✅ Auto-initialization completed successfully!")
        else:
            logger.error("❌ Auto-initialization failed - will retry on first request")
    else:
        logger.info("⏳ Hybrid SadTalker will be initialized on first request")
    
    logger.info("✅ Hybrid SadTalker Service Ready!")
    logger.info("🎯 Best of Both Worlds: Your API + Official Quality!")

# Run startup
startup()

# Start RunPod serverless worker
logger.info("🌐 Starting RunPod with Hybrid SadTalker...")
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})