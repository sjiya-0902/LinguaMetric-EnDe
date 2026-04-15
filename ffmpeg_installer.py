import os
import sys
import shutil
import zipfile
import urllib.request
import platform

def install_ffmpeg_locally(target_dir="ffmpeg_bin"):
    # Check if already installed globally
    if shutil.which("ffmpeg"):
        return None

    # Check if already installed locally
    local_bin = os.path.abspath(os.path.join(target_dir, "bin"))
    ffmpeg_exe = os.path.join(local_bin, "ffmpeg.exe")
    if os.path.exists(ffmpeg_exe):
        return local_bin

    print("ffmpeg not found. Downloading portable version...")
    
    os.makedirs(target_dir, exist_ok=True)
    
    # URL for Windows build (gyan.dev) - using a specific release for stability
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = os.path.join(target_dir, "ffmpeg.zip")
    
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        # The zip usually contains a subfolder like 'ffmpeg-6.0-essentials_build'
        extracted_folders = [f for f in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, f))]
        for folder in extracted_folders:
            potential_bin = os.path.join(target_dir, folder, "bin")
            if os.path.exists(os.path.join(potential_bin, "ffmpeg.exe")):
                # Move contents of bin to our target local_bin or just return this path
                return os.path.abspath(potential_bin)
                
    except Exception as e:
        print(f"Failed to download/install ffmpeg: {e}")
        return None
    
    return None

def setup_ffmpeg_path():
    ffmpeg_bin = install_ffmpeg_locally()
    if ffmpeg_bin:
        print(f"Adding {ffmpeg_bin} to PATH")
        os.environ["PATH"] += os.pathsep + ffmpeg_bin
        return True
    elif shutil.which("ffmpeg"):
        return True
    else:
        return False