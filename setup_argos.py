# setup_argos.py (Improved and more robust version)
import argostranslate.package
import argostranslate.translate
import sys

def main():
    """
    This script downloads and installs the required Argos Translate
    language package for Japanese to English translation.
    """
    try:
        print("Updating Argos Translate package index...")
        argostranslate.package.update_package_index()
        print("Index updated successfully.")
    except Exception as e:
        print(f"\nERROR: Failed to update the package index.", file=sys.stderr)
        print("This is likely a network issue. Please check your internet connection/firewall.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    available_packages = argostranslate.package.get_available_packages()
    
    package_to_install = next(
        filter(
            lambda x: x.from_code == "ja" and x.to_code == "en", available_packages
        ),
        None
    )
    
    if package_to_install:
        print("\nFound Japanese to English package. Downloading and installing...")
        print("This may take a minute or two...")
        try:
            package_to_install.install()
            print("Package installed successfully.")
        except Exception as e:
            print(f"\nERROR: Failed to download or install the package.", file=sys.stderr)
            print("This could be a network issue or a file permissions problem.", file=sys.stderr)
            print(f"Details: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("\nCould not find the Japanese to English package in the updated index.")
        print("This is unusual. Let's check if it's already installed.")

    # --- Verification Step ---
    print("\nVerifying installation...")
    try:
        installed_languages = argostranslate.translate.get_installed_languages()
        
        ja_lang_found = any(lang.code == "ja" for lang in installed_languages)
        en_lang_found = any(lang.code == "en" for lang in installed_languages)

        if ja_lang_found and en_lang_found:
            print("\n✅ Verification successful: Japanese and English models are installed.")
            print("You can now run the main 'run.py' application.")
        else:
            print("\n❌ Verification failed.", file=sys.stderr)
            if not ja_lang_found:
                print("- Japanese model is missing.", file=sys.stderr)
            if not en_lang_found:
                print("- English model is missing.", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred during verification.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()