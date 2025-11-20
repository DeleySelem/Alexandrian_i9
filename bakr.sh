#!/bin/bash
# BlackArch Keyring Installation Script
# Requires root privileges

install_blackarch_keyring() {
    # Remove existing keyring to prevent conflicts
    apt purge -y blackarch-keyring &> /dev/null
    rm -rf /etc/apt/trusted.gpg.d/blackarch.gpg* &> /dev/null

    # Install essential dependencies
    apt update && apt install -y gnupg wget

    # Download and install BlackArch keyring (official method)
    wget https://blackarch.org/keyring/blackarch-keyring.pkg.tar.xz{,.sig}

    # Verify PGP signature (security critical)
    if ! gpg --keyserver hkps://keyserver.ubuntu.com --recv-keys 4345771566D76038; then
        echo "❌ Error: Failed to import BlackArch PGP key" >&2
        exit 1
    fi

    if ! gpg --verify blackarch-keyring.pkg.tar.xz.sig blackarch-keyring.pkg.tar.xz; then
        echo "❌ Critical: PGP signature verification failed!" >&2
        exit 1
    fi

    # Extract and install keyring
    tar -xJf blackarch-keyring.pkg.tar.xz
    cp -f usr/share/pacman/keyrings/blackarch{.gpg,-trusted} /etc/apt/trusted.gpg.d/
    chmod 644 /etc/apt/trusted.gpg.d/blackarch{.gpg,-trusted}

    # Cleanup temporary files
    rm -rf blackarch-keyring.* usr

    # Verify installation
    if apt-key finger | grep -q '4345 7715 66D7 6038'; then
        echo "✅ BlackArch keyring installed successfully"
    else
        echo "❌ Installation verification failed" >&2
        exit 1
    fi
}

# Security hardening
security_hardening() {
    # Configure APT security
    echo 'Acquire::AllowInsecureRepositories "false";' > /etc/apt/apt.conf.d/99-security.conf
    echo 'APT::Sandbox::User "root";' >> /etc/apt/apt.conf.d/99-security.conf

    # Set keyring permissions
    chmod 0644 /etc/apt/trusted.gpg.d/blackarch{.gpg,-trusted}
    chown root:root /etc/apt/trusted.gpg.d/blackarch{.gpg,-trusted}

    # Configure GPG strict mode
    echo "keyserver hkps://keyserver.ubuntu.com" >> /etc/gnupg/gpg.conf
    echo "keyserver-options no-honor-keyserver-url" >> /etc/gnupg/gpg.conf
}

# Repository configuration (optional)
configure_repositories() {
    if [ ! -f /etc/apt/sources.list.d/blackarch.list ]; then
        cat > /etc/apt/sources.list.d/blackarch.list <<- EOL
# BlackArch repository
deb [signed-by=/etc/apt/trusted.gpg.d/blackarch.gpg] https://mirror.cyberbits.eu/blackarch blackarch main
deb [signed-by=/etc/apt/trusted.gpg.d/blackarch.gpg] https://blackarch.org/repo blackarch main
EOL
        echo "ℹ️ BlackArch repositories added (commented by default)"
        echo "   Uncomment in /etc/apt/sources.list.d/blackarch.list to enable"
    fi
}

# Main execution
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root. Use sudo!" >&2
    exit 1
fi

install_blackarch_keyring
security_hardening
configure_repositories

echo "--------------------------------------------------"
echo "BlackArch Keyring Setup Complete"
echo "To install BlackArch tools:"
echo "1. Enable repositories in: /etc/apt/sources.list.d/blackarch.list"
echo "2. Run: sudo apt update && sudo apt install <blackarch-tool>"
echo "--------------------------------------------------"
