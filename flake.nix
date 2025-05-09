{
    description = "Python nix shell";
# May run nix develop under any subdirectory of this

    inputs = {
        flake-utils.url = "github:numtide/flake-utils";
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
        nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";

    }; 

    outputs = { self, nixpkgs, flake-utils, ... }:
        flake-utils.lib.eachDefaultSystem
        (system:
        let pkgs = nixpkgs.legacyPackages.${system}; in
            {
                # python shell with nix packaged python packages
                devShells.default = import ./shell.nix { inherit pkgs; };  
            }
        );
}
