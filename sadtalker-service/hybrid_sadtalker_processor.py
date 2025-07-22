import os
import subprocess
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class HybridSadTalkerProcessor:
    """Hybrid SadTalker Processor - API Wrapper + Official inference.py"""
    
    def __init__(self, device, model_path):
        self.device = device
        self.model_path = model_path
        self.sadtalker_root = '/app/SadTalker'
        self.inference_script = os.path.join(self.sadtalker_root, 'inference.py')
        
        # Default settings for maximum quality
        self.default_settings = {
            'still': True,
            'enhancer': 'gfpgan',
            'preprocess': 'full',
            'size': 512,
            'expression_scale': 1.0,
            'pose_style': 0
        }
        
        # Verify official inference.py exists
        self.verify_official_setup()
        
        logger.info("✅ Hybrid SadTalker Processor initialized (API + Official)")

    def verify_official_setup(self):
        """Verify official SadTalker setup"""
        try:
            if not os.path.exists(self.sadtalker_root):
                raise Exception(f"Official SadTalker not found at: {self.sadtalker_root}")
            
            if not os.path.exists(self.inference_script):
                raise Exception(f"Official inference.py not found at: {self.inference_script}")
            
            # Check if models are available
            checkpoints_dir = os.path.join(self.model_path, 'checkpoints')
            if not os.path.exists(checkpoints_dir):
                raise Exception(f"Model checkpoints not found at: {checkpoints_dir}")
            
            logger.info(f"✅ Official SadTalker verified: {self.sadtalker_root}")
            logger.info(f"✅ Inference script verified: {self.inference_script}")
            logger.info(f"✅ Model path verified: {checkpoints_dir}")
            
        except Exception as e:
            logger.error(f"❌ Official SadTalker setup verification failed: {e}")
            raise

    def apply_enhancement_settings(self, custom_settings):
        """Apply custom settings (merged with defaults)"""
        try:
            settings = self.default_settings.copy()
            if custom_settings:
                settings.update(custom_settings)
            
            logger.info(f"🔧 Applied settings: {settings}")
            return settings
            
        except Exception as e:
            logger.error(f"❌ Failed to apply settings: {e}")
            return self.default_settings

    def build_inference_command(self, image_path, audio_path, output_dir, settings):
        """Build official inference.py command with all parameters"""
        try:
            cmd = [
                'python', self.inference_script,
                '--driven_audio', audio_path,
                '--source_image', image_path,
                '--result_dir', output_dir,
                '--checkpoint_dir', os.path.join(self.model_path, 'checkpoints'),
                '--size', str(settings.get('size', 512)),
                '--pose_style', str(settings.get('pose_style', 0)),
                '--expression_scale', str(settings.get('expression_scale', 1.0))
            ]
            
            # Add conditional flags
            if settings.get('still', True):
                cmd.append('--still')
            
            if settings.get('enhancer') == 'gfpgan':
                cmd.append('--enhancer')
                cmd.append('gfpgan')
            
            if settings.get('preprocess') == 'full':
                cmd.append('--preprocess')
                cmd.append('full')
            
            logger.info(f"🔧 Built command: {' '.join(cmd)}")
            return cmd
            
        except Exception as e:
            logger.error(f"❌ Failed to build inference command: {e}")
            raise

    def process_talking_head(self, image_path, audio_path, output_path, settings=None):
        """Main processing using Official SadTalker via subprocess"""
        try:
            logger.info("🎭 Starting Hybrid SadTalker Processing...")
            logger.info(f"📷 Image: {image_path}")
            logger.info(f"🎵 Audio: {audio_path}")
            logger.info(f"🎬 Output: {output_path}")
            
            # Apply settings
            final_settings = self.apply_enhancement_settings(settings)
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_output_dir:
                logger.info(f"📁 Temporary output: {temp_output_dir}")
                
                # Build official inference command
                cmd = self.build_inference_command(
                    image_path, audio_path, temp_output_dir, final_settings
                )
                
                # Set environment variables for official SadTalker
                env = os.environ.copy()
                env['PYTHONPATH'] = f"/app/SadTalker:{env.get('PYTHONPATH', '')}"
                env['CUDA_VISIBLE_DEVICES'] = '0' if self.device == 'cuda' else ''
                
                # Execute official inference.py
                logger.info("🚀 Executing Official SadTalker inference.py...")
                
                start_time = datetime.now()
                result = subprocess.run(
                    cmd,
                    cwd=self.sadtalker_root,  # Run from SadTalker directory
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                end_time = datetime.now()
                
                processing_time = (end_time - start_time).total_seconds()
                logger.info(f"⏰ Official processing completed in: {processing_time:.1f}s")
                
                # Check if execution was successful
                if result.returncode != 0:
                    logger.error(f"❌ Official inference failed with return code: {result.returncode}")
                    logger.error(f"❌ STDERR: {result.stderr}")
                    logger.error(f"❌ STDOUT: {result.stdout}")
                    raise Exception(f"Official SadTalker failed: {result.stderr}")
                
                logger.info("✅ Official SadTalker executed successfully")
                if result.stdout:
                    logger.info(f"📝 Output: {result.stdout[:500]}...")  # First 500 chars
                
                # Find generated video file
                generated_video = self.find_generated_video(temp_output_dir)
                if not generated_video:
                    raise Exception("Generated video not found in output directory")
                
                # Move to final output location
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.move(generated_video, output_path)
                
                # Verify final output
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)
                    logger.info(f"🎉 Hybrid processing completed successfully!")
                    logger.info(f"📊 Final output: {output_path} ({file_size:.1f}MB)")
                    
                    return {
                        'success': True,
                        'video_path': output_path,
                        'processing_time': processing_time,
                        'settings_used': final_settings,
                        'file_size_mb': file_size,
                        'architecture': 'Hybrid_API_Official_SadTalker',
                        'quality_level': 'Official_Maximum_Quality'
                    }
                else:
                    raise Exception("Final output file not created")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Official SadTalker processing timed out")
            return {
                'success': False,
                'error': 'Processing timed out after 5 minutes'
            }
        except Exception as e:
            logger.error(f"❌ Hybrid processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def find_generated_video(self, output_dir):
        """Find the generated video file in output directory"""
        try:
            # SadTalker creates timestamped directories with .mp4 files
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.mp4'):
                        video_path = os.path.join(root, file)
                        logger.info(f"🎬 Found generated video: {video_path}")
                        return video_path
            
            logger.error("❌ No .mp4 file found in output directory")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error finding generated video: {e}")
            return None

    def get_model_status(self):
        """Get status of hybrid setup"""
        status = {
            'hybrid_mode': True,
            'official_inference_available': os.path.exists(self.inference_script),
            'sadtalker_root': self.sadtalker_root,
            'model_path': self.model_path,
            'device': str(self.device),
            'default_settings': self.default_settings,
            'architecture': 'Hybrid_API_Wrapper_Official_SadTalker'
        }
        
        # Check model files
        checkpoints_dir = os.path.join(self.model_path, 'checkpoints')
        if os.path.exists(checkpoints_dir):
            model_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(('.pth', '.tar'))]
            status['available_models'] = len(model_files)
            status['model_files'] = model_files[:5]  # First 5 models
        
        return status