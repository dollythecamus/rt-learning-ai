{ pkgs ? import <nixpkgs> {}, unstable ? import <nixos-unstable> {}}:
pkgs.mkShell {
  name = "slowsophy-env";

  buildInputs = [
    pkgs.python312
    pkgs.python312Packages.pygame
    pkgs.python312Packages.numpy
    pkgs.python312Packages.torch
    pkgs.python312Packages.matplotlib
    
    
    ];

  shellHook = ''
    echo "welcome, this is yurigochi"
  '';
}

