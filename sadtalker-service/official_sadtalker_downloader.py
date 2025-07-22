import os
import requests
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

def download_official_sadtalker_models(model_path):
    """Download Official SadTalker models using official script"""
    try:
        logger.info("🎭 Downloading Official SadTalker Models...")
        
        # Create model directories
        os.makedirs(model_path, exist_ok=True)
        checkpoints_dir = os.path.join(model_path, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Use official SadTalker download script
        sadtalker_root = '/app/SadTalker'
        download_script = os.path.join(sadtalker_root, 'scripts', 'download_models.sh')
        
        if os.path.exists(download_script):
            logger.info("📥 Using official download script...")
            
            # Change to SadTalker directory and run official script
            original_dir = os.getcwd()
            try:
                os.chdir(sadtalker_root)
                
                # Run official download script
                result = subprocess.run(['bash', 'scripts/download_models.sh'], 
                                      capture_output=True, text=True, timeout=1800)
                
                if result.returncode == 0:
                    logger.info("✅ Official models downloaded successfully")
                    
                    # Move models to our model path if different
                    official_checkpoints = os.path.join(sadtalker_root, 'checkpoints')
                    if official_checkpoints != checkpoints_dir and os.path.exists(official_checkpoints):
                        logger.info(f"📁 Moving models from {official_checkpoints} to {checkpoints_dir}")
                        import shutil
                        
                        # Copy all files from official checkpoints to our location
                        for item in os.listdir(official_checkpoints):
                            src = os.path.join(official_checkpoints, item)
                            dst = os.path.join(checkpoints_dir, item)
                            if os.path.isfile(src):
                                shutil.copy2(src, dst)
                            elif os.path.isdir(src):
                                shutil.copytree(src, dst, dirs_exist_ok=True)
                    
                    return True
                else:
                    logger.error(f"❌ Official download script failed: {result.stderr}")
                    # Fallback to manual download
                    return download_manual_models(checkpoints_dir)
                    
            finally:
                os.chdir(original_dir)
        else:
            logger.warning("⚠️ Official download script not found, using manual download")
            return download_manual_models(checkpoints_dir)
            
    except Exception as e:
        logger.error(f"❌ Official model download failed: {e}")
        return download_manual_models(checkpoints_dir)

def download_manual_models(checkpoints_dir):
    """Manual download of critical SadTalker models"""
    try:
        logger.info("📥 Downloading models manually...")
        
        # Critical models for 512px SadTalker
        models = {
            # 512px Models (New Version)
            "SadTalker_V0.0.2_512.safetensors": {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors",
                "size_mb": 400
            },
            
            # Core Models (if safetensors doesn't work)
            "auido2exp_00300-model.pth": {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.1/auido2exp_00300-model.pth",
                "size_mb": 85
            },
            "auido2pose_00140-model.pth": {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.1/auido2pose_00140-model.pth",
                "size_mb": 85
            },
            "mapping_00229-model.pth.tar": {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.1/mapping_00229-model.pth.tar",
                "size_mb": 30
            },
            "facevid2vid_00189-model.pth.tar": {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.1/facevid2vid_00189-model.pth.tar",
                "size_mb": 120
            },
            "epoch_20.pth": {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.1/epoch_20.pth",
                "size_mb": 200
            },
            "wav2lip.pth": {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.1/wav2lip.pth",
                "size_mb": 44
            }
        }
        
        success_count = 0
        total_models = len(models)
        
        for model_name, config in models.items():
            try:
                model_path = os.path.join(checkpoints_dir, model_name)
                
                # Skip if exists and is valid size
                if os.path.exists(model_path):
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    if size_mb > config["size_mb"] * 0.8:  # At least 80% of expected size
                        logger.info(f"✅ {model_name} already exists ({size_mb:.1f}MB)")
                        success_count += 1
                        continue
                
                logger.info(f"📥 Downloading {model_name}...")
                
                response = requests.get(config["url"], stream=True, timeout=300)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify download
                if os.path.exists(model_path):
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    if size_mb > config["size_mb"] * 0.5:  # At least 50% of expected size
                        logger.info(f"✅ {model_name} downloaded ({size_mb:.1f}MB)")
                        success_count += 1
                    else:
                        logger.error(f"❌ {model_name} too small ({size_mb:.1f}MB)")
                        os.remove(model_path)
                
            except Exception as e:
                logger.error(f"❌ Failed to download {model_name}: {e}")
                continue
        
        logger.info(f"📊 Models downloaded: {success_count}/{total_models}")
        
        # Need at least the safetensors model OR the core models
        if success_count >= 1:
            return True
        else:
            logger.error("❌ Failed to download sufficient models")
            return False
            
    except Exception as e:
        logger.error(f"❌ Manual model download failed: {e}")
        return False

def verify_sadtalker_models(model_path):
    """Verify SadTalker models exist"""
    try:
        checkpoints_dir = os.path.join(model_path, 'checkpoints')
        
        if not os.path.exists(checkpoints_dir):
            logger.error(f"❌ Checkpoints directory not found: {checkpoints_dir}")
            return False
        
        # Check for safetensors model (preferred)
        safetensors_512 = os.path.join(checkpoints_dir, 'SadTalker_V0.0.2_512.safetensors')
        if os.path.exists(safetensors_512) and os.path.getsize(safetensors_512) > 100 * 1024 * 1024:
            logger.info("✅ 512px SafeTensors model found")
            return True
        
        # Check for core models
        core_models = [
            'auido2exp_00300-model.pth',
            'auido2pose_00140-model.pth',
            'mapping_00229-model.pth.tar',
            'facevid2vid_00189-model.pth.tar',
            'epoch_20.pth'
        ]
        
        found_models = 0
        for model in core_models:
            model_path_full = os.path.join(checkpoints_dir, model)
            if os.path.exists(model_path_full) and os.path.getsize(model_path_full) > 1024 * 1024:
                found_models += 1
                logger.info(f"✅ Found: {model}")
            else:
                logger.warning(f"⚠️ Missing: {model}")
        
        if found_models >= 4:  # Need at least 4/5 core models
            logger.info(f"✅ Core models verified: {found_models}/5")
            return True
        else:
            logger.error(f"❌ Insufficient models: {found_models}/5")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model verification failed: {e}")
        return False