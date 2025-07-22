# Official SadTalker RunPod Application - Complete Implementation
# Based on True OpenTalker/SadTalker Architecture

from official_sadtalker_downloader import download_official_sadtalker_models, verify_sadtalker_models
from official_sadtalker_processor_real import OfficialSadTalkerProcessor
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
official_sadtalker = None
device = None

def get_storage_path():
    """Detect and return the network volume mount path"""
    # Check environment variable first
    if "RUNPOD_VOLUME_PATH" in os.environ:
        env_path = os.environ["RUNPOD_VOLUME_PATH"]
        if os.path.exists(env_path) and os.access(env_path, os.W_OK):
            logger.info(f"📁 Using environment volume path: {env_path}")
            return env_path
    
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
    os.makedirs("/tmp/official_sadtalker", exist_ok=True)
    logger.warning("⚠️ Using fallback storage: /tmp/official_sadtalker")
    return "/tmp/official_sadtalker"

def get_model_storage_path():
    """Get dedicated model storage path for official SadTalker"""
    base_path = get_storage_path()
    model_path = os.path.join(base_path, "official_sadtalker", "models")
    os.makedirs(model_path, exist_ok=True)
    logger.info(f"📂 Official SadTalker models path: {model_path}")
    return model_path

def initialize_official_sadtalker():
    """Initialize Complete Official SadTalker with all components"""
    global official_sadtalker, device
    
    try:
        logger.info("🚀 Starting Complete Official SadTalker Initialization...")
        
        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"📱 Using device: {device}")
        
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🔥 GPU: {gpu_name}")
                logger.info(f"💾 GPU Memory: {gpu_memory:.1f} GB")
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                logger.info("🧹 GPU cache cleared")
            except Exception as e:
                logger.warning(f"⚠️ GPU info failed: {e}")
        
        # Get model storage path
        model_path = get_model_storage_path()
        logger.info(f"📂 Model storage path: {model_path}")
        
        # Check and download official models
        logger.info("🔍 Checking official SadTalker models...")
        if not verify_sadtalker_models(model_path):
            logger.info("📥 Official models not found, downloading...")
            
            try:
                download_success = download_official_sadtalker_models(model_path)
                if not download_success:
                    logger.error("❌ Failed to download official SadTalker models")
                    return False
                
                # Verify after download
                logger.info("🔍 Verifying models after download...")
                if not verify_sadtalker_models(model_path):
                    logger.error("❌ Model verification failed after download")
                    return False
                else:
                    logger.info("✅ Models verified successfully after download")
                    
            except Exception as e:
                logger.error(f"❌ Download process failed: {e}")
                import traceback
                logger.error(f"🔍 Download traceback: {traceback.format_exc()}")
                return False
        else:
            logger.info("✅ All official SadTalker models verified")
        
        # Initialize Official SadTalker Processor
        logger.info("📦 Loading Complete Official SadTalker Processor...")
        try:
            processor = OfficialSadTalkerProcessor(device, model_path)
            logger.info("✅ OfficialSadTalkerProcessor created successfully")
            
            # Get model status
            model_status = processor.get_model_status()
            logger.info(f"📊 Model Status: {model_status}")
            
        except Exception as e:
            logger.error(f"❌ OfficialSadTalkerProcessor initialization failed: {e}")
            import traceback
            logger.error(f"🔍 Processor traceback: {traceback.format_exc()}")
            return False
        
        # Update global model variable
        official_sadtalker = {
            "initialized": True,
            "device": device,
            "ready": True,
            "model_path": model_path,
            "processor": processor,
            "version": "Official_SadTalker_Complete_v1.0",
            "architecture": "CropAndExtract+Audio2Pose+Audio2Exp+FaceVid2Vid+MappingNet",
            "model_status": model_status
        }
        
        logger.info("🎉 Complete Official SadTalker loaded successfully!")
        logger.info(f"🎭 Architecture: {official_sadtalker['architecture']}")
        logger.info(f"🔧 Components: Deep3DFaceReconstruction + PoseVAE + ExpNet + Face-vid2vid + MappingNet")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Official SadTalker: {str(e)}")
        import traceback
        logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
        return False

def validate_inputs(image_data, audio_data):
    """Validate input image and audio data"""
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

def generate_official_talking_head(image_base64, audio_base64, settings=None):
    """Generate talking head using Complete Official SadTalker"""
    global official_sadtalker, device
    
    try:
        logger.info("🎭 Starting Complete Official SadTalker Professional Generation...")
        
        # LOG SETTINGS
        if settings:
            logger.info(f"🔧 Official settings received: {settings}")
            logger.info(f"🎯 Still mode: {settings.get('still', False)}")
            logger.info(f"✨ Enhancer: {settings.get('enhancer', 'none')}")
            logger.info(f"📊 Expression scale: {settings.get('expression_scale', 1.0)}")
            logger.info(f"🎨 Preprocess: {settings.get('preprocess', 'crop')}")
            logger.info(f"📐 Size: {settings.get('size', 512)}")
        else:
            logger.info("⚠️ No settings provided - using official defaults")
        
        if official_sadtalker is None:
            raise Exception("Official SadTalker not initialized")
        
        # Get storage path
        storage_path = get_storage_path()
        
        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()
            logger.info("🧹 GPU cache cleared")
        
        # Decode inputs
        logger.info("🔄 Processing inputs...")
        try:
            # Clean base64 strings
            image_base64_clean = image_base64.strip().replace('\n', '').replace('\r', '')
            audio_base64_clean = audio_base64.strip().replace('\n', '').replace('\r', '')
            
            # Decode
            image_data = base64.b64decode(image_base64_clean)
            audio_data = base64.b64decode(audio_base64_clean)
            
            logger.info(f"📷 Image decoded: {len(image_data)} bytes")
            logger.info(f"🎵 Audio decoded: {len(audio_data)} bytes")
            
            # Validate data sizes
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
        output_video_path = os.path.join(storage_path, f"official_result_{timestamp}.mp4")
        
        try:
            # Save image
            with open(input_image_path, 'wb') as f:
                f.write(image_data)
            logger.info(f"📷 Image saved: {input_image_path}")
            
            # Validate image
            test_image = cv2.imread(input_image_path)
            if test_image is None:
                raise Exception(f"Invalid image file: {input_image_path}")
            logger.info(f"✅ Image validation passed: {test_image.shape}")
            
            # Save audio
            with open(input_audio_path, 'wb') as f:
                f.write(audio_data)
            logger.info(f"🎵 Audio saved: {input_audio_path}")
            
            # Validate audio
            if not os.path.exists(input_audio_path) or os.path.getsize(input_audio_path) < 1000:
                raise Exception(f"Invalid audio file: {input_audio_path}")
            logger.info(f"✅ Audio validation passed: {os.path.getsize(input_audio_path)} bytes")
            
        except Exception as e:
            raise Exception(f"Failed to save input files: {str(e)}")
        
        # Get Official SadTalker processor
        processor = official_sadtalker["processor"]
        
        # Apply enhancement settings
        if settings:
            logger.info("🔧 Applying official enhancement settings...")
            settings_applied = processor.apply_enhancement_settings(settings)
            if settings_applied:
                logger.info("✅ Official enhancement settings applied successfully")
            else:
                logger.warning("⚠️ Failed to apply some enhancement settings")
        else:
            logger.info("ℹ️ Using default official SadTalker settings")
        
        # Process using Complete Official SadTalker pipeline
        logger.info("🎬 Processing with Complete Official SadTalker Pipeline...")
        logger.info("📋 Pipeline: CropAndExtract → Audio2Pose+Audio2Exp → AnimateFromCoeff")
        
        result = processor.process_talking_head(
            input_image_path, 
            input_audio_path, 
            output_video_path
        )
        
        if not result['success']:
            raise Exception(f"Official SadTalker processing failed: {result.get('error', 'Unknown error')}")
        
        # Verify output
        if not os.path.exists(output_video_path):
            raise Exception("Output video not created")
        
        file_size = os.path.getsize(output_video_path)
        
        logger.info("🎉 Complete Official SadTalker generation completed successfully!")
        logger.info(f"🎭 Face Model: {result.get('face_model_used', 'Official_Deep3DFaceReconstruction')}")
        logger.info(f"🧠 Networks: {result.get('networks_used', 'Official_Complete_Pipeline')}")
        logger.info(f"🔧 Enhancements: {result.get('enhancement_settings', {})}")
        
        return {
            "success": True,
            "video_path": output_video_path,
            "video_filename": os.path.basename(output_video_path),
            "video_size_mb": file_size / (1024 * 1024),
            "duration_seconds": result.get('duration', 0),
            "file_size_bytes": file_size,
            "storage_location": storage_path,
            "input_image_path": input_image_path,
            "input_audio_path": input_audio_path,
            "model_path": official_sadtalker["model_path"],
            "lip_sync_enabled": True,
            "face_model_used": result.get('face_model_used', 'Official_Deep3DFaceReconstruction'),
            "networks_used": result.get('networks_used', 'Complete_Official_Pipeline'),
            "architecture": result.get('architecture', 'Official_SadTalker_Complete'),
            "sadtalker_version": official_sadtalker.get("version", "Official_Complete_v1.0"),
            "quality_level": "Professional_Official_Grade",
            "enhancement_settings": result.get('enhancement_settings', {}),
            "crop_info": result.get('crop_info', {}),
            "model_status": official_sadtalker.get("model_status", {})
        }
        
    except Exception as e:
        logger.error(f"❌ Official SadTalker generation failed: {str(e)}")
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return {
            "success": False,
            "error": str(e)
        }

def handler(event):
    """Main handler for Official SadTalker requests"""
    try:
        logger.info(f"📨 Received official SadTalker event: {event}")
        
        # Extract input from RunPod format
        input_data = event.get("input", {})
        image_base64 = input_data.get("image")
        audio_base64 = input_data.get("audio")
        settings = input_data.get("settings", {})
        
        # Log received settings
        if settings:
            logger.info(f"🔧 Received official settings: {settings}")
        
        # Validation
        is_valid, validation_msg = validate_inputs(image_base64, audio_base64)
        if not is_valid:
            logger.error(f"❌ Validation failed: {validation_msg}")
            yield {"error": validation_msg}
            return
        
        logger.info("🎯 Processing Complete Official SadTalker generation...")
        
        # Yield progress - initialization
        yield {
            "status": "initializing",
            "message": "Starting Complete Official SadTalker generation...",
            "progress": "10%",
            "architecture": "Complete_Official_SadTalker_Pipeline"
        }
        
        # Check if model is initialized
        if official_sadtalker is None:
            logger.info("🔄 Model not initialized, initializing now...")
            yield {
                "status": "loading_model", 
                "message": "Loading Complete Official SadTalker models...",
                "progress": "20%"
            }
            
            success = initialize_official_sadtalker()
            if not success:
                logger.error("❌ Official model initialization failed")
                yield {
                    "error": "Failed to initialize Complete Official SadTalker",
                    "status": "failed"
                }
                return
        
        # Yield progress - processing
        yield {
            "status": "processing",
            "message": "Processing with Complete Official Pipeline (CropAndExtract → Audio2Coeff → AnimateFromCoeff)...",
            "progress": "50%"
        }
        
        # Generate talking head
        start_time = datetime.now()
        logger.info(f"⏰ Starting Official generation at: {start_time}")
        
        result = generate_official_talking_head(
            image_base64=image_base64,
            audio_base64=audio_base64,
            settings=settings
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.info(f"⏰ Official generation completed in: {processing_time:.1f}s")
        
        if result["success"]:
            # Yield progress - saving
            yield {
                "status": "saving",
                "message": "Saving Official SadTalker video...",
                "progress": "90%"
            }
            
            # FINAL YIELD - Success
            logger.info(f"✅ Official talking head generated in {processing_time:.1f}s")
            logger.info(f"📊 Video size: {result['video_size_mb']:.2f} MB")
            logger.info(f"📁 Video saved: {result['video_path']}")
            logger.info(f"🎭 Architecture: {result['architecture']}")
            
            # S3 access info
            s3_endpoint = "https://s3api-eu-ro-1.runpod.io/"
            volume_name = "pavidwplk8"
            s3_object_key = result["video_filename"]
            
            yield {
                "video_filename": result["video_filename"],
                "video_path": result["video_path"],
                "status": "completed",
                "success": True,
                "progress": "100%",
                "architecture": result["architecture"],
                "s3_access": {
                    "endpoint": s3_endpoint,
                    "bucket": volume_name,
                    "object_key": s3_object_key,
                    "s3_url": f"s3://{volume_name}/{s3_object_key}",
                    "download_instruction": "Use AWS CLI with RunPod S3 credentials"
                },
                "metadata": {
                    "model": "Complete_Official_SadTalker",
                    "architecture": result["architecture"], 
                    "version": result["sadtalker_version"],
                    "face_model_used": result["face_model_used"],
                    "networks_used": result["networks_used"],
                    "enhancement_settings": result["enhancement_settings"],
                    "duration_seconds": result["duration_seconds"],
                    "video_size_mb": round(result["video_size_mb"], 2),
                    "processing_time_seconds": round(processing_time, 1),
                    "gpu_available": torch.cuda.is_available(),
                    "device": str(device),
                    "quality_level": result.get("quality_level", "Professional_Official"),
                    "timestamp": datetime.now().isoformat(),
                    "model_status": result.get("model_status", {})
                }
            }
            
        else:
            # Generation failed
            logger.error(f"❌ Official generation failed: {result['error']}")
            yield {
                "error": result["error"],
                "status": "failed",
                "success": False,
                "processing_time_seconds": round(processing_time, 1),
                "timestamp": datetime.now().isoformat(),
                "architecture": "Official_SadTalker_Failed"
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
    """Startup function - initialize complete official system"""
    logger.info("🚀 Starting Complete Official SadTalker Service...")
    logger.info(f"🎭 Version: Complete Official SadTalker Professional v1.0")
    logger.info(f"🏗️ Architecture: CropAndExtract + Audio2Pose + Audio2Exp + FaceVid2Vid + MappingNet")
    logger.info(f"🐍 Python version: {os.sys.version}")
    logger.info(f"🔥 PyTorch version: {torch.__version__}")
    logger.info(f"💻 CUDA available: {torch.cuda.is_available()}")
    
    # Check storage
    storage_path = get_storage_path()
    logger.info(f"📁 Storage initialized: {storage_path}")
    
    # Create output directory
    output_dir = os.path.join(storage_path, "official_sadtalker_output")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"📂 Output directory: {output_dir}")
    
    # Model storage
    model_path = get_model_storage_path()
    logger.info(f"🎭 Model storage: {model_path}")
    
    # GPU info
    if torch.cuda.is_available():
        logger.info(f"🎮 CUDA version: {torch.version.cuda}")
        logger.info(f"🔧 GPU device: {torch.cuda.get_device_name(0)}")
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"💾 Total GPU memory: {total_memory:.1f} GB")
    
    # Auto-initialize if requested
    auto_init = os.getenv("AUTO_INIT_MODEL", "false").lower() == "true"
    if auto_init:
        logger.info("🔄 AUTO_INIT_MODEL enabled, initializing Official SadTalker...")
        success = initialize_official_sadtalker()
        if success:
            logger.info("✅ Auto-initialization completed successfully!")
            if official_sadtalker:
                logger.info(f"🎭 Architecture: {official_sadtalker.get('architecture', 'Unknown')}")
                logger.info(f"📊 Model Status: {official_sadtalker.get('model_status', {})}")
        else:
            logger.error("❌ Auto-initialization failed - will retry on first request")
    else:
        logger.info("⏳ Official SadTalker will be initialized on first request")
    
    logger.info("✅ Complete Official SadTalker service startup completed!")
    logger.info("🎯 Ready to generate professional-quality talking heads with complete official architecture!")

# Run startup
startup()

# Start RunPod serverless worker
logger.info("🌐 Starting RunPod serverless worker with Complete Official SadTalker...")
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})