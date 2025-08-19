#!/usr/bin/env python3
import subprocess
import sys

def main():
    """Run the Streamlit app"""
    try:
        # Set the theme and config
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--theme.primaryColor", "#0066cc",
            "--theme.backgroundColor", "#f0f2f6",
            "--theme.secondaryBackgroundColor", "#ffffff",
            "--theme.textColor", "#262730",
            "--theme.font", "sans-serif"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main() 