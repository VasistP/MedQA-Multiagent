# setup.py
import os
import sys
from dotenv import load_dotenv


def check_setup():
    """Check if environment is properly set up"""

    print("Checking environment setup...")

    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    print("✅ Python version OK")

    # Check OpenAI API key
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Please create a .env file with: OPENAI_API_KEY=your_key_here")
        return False
    print("✅ OpenAI API key found")

    # Check wandb
    try:
        import wandb
        if not os.getenv('WANDB_API_KEY'):
            print("⚠️  WANDB_API_KEY not set - you'll need to login interactively")
            print("   Run: wandb login")
        else:
            print("✅ Wandb API key found")
    except ImportError:
        print("❌ wandb not installed")
        return False

    # Check data directory
    if not os.path.exists("./data/medqa"):
        print("⚠️  MedQA dataset not found")
        print("   Run: python src/utils/download_dataset.py")
    else:
        print("✅ MedQA dataset found")

    return True


if __name__ == "__main__":
    if check_setup():
        print("\n✅ Environment setup complete!")
    else:
        print("\n❌ Please fix the issues above before proceeding")
