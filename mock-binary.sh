#!/bin/bash

# Create a simple binary for Docker testing
cat > target/release/arthachain.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> // For sleep function

int main(int argc, char *argv[]) {
    printf("Artha Chain Node v0.1.0 (Mock)\n");
    printf("Command line arguments:\n");
    
    for (int i = 1; i < argc; i++) {
        printf("  %s\n", argv[i]);
    }
    
    // If API enabled, start a simple server
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--enable-api") == 0) {
            printf("API server started at 0.0.0.0:3000\n");
            
            // Loop forever to keep the container running
            while(1) {
                sleep(60);
            }
        }
    }
    
    return 0;
}
EOF

# Compile the mock binary
gcc -o target/release/arthachain target/release/arthachain.c

# Make it executable
chmod +x target/release/arthachain

echo "Mock binary created at target/release/arthachain" 