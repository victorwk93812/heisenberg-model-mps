{pkgs ? import <nixpkgs> {} }: 
with pkgs;
mkShell {
    # Using packages instead of buildInputs is also fine
    packages = [ 
        # nixpkgs-fmt
        (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
# select Python packages here
            numpy
            # scipy
            # pandas
            # pyqt5
            # pyqt6
            # matplotlib
            # uncertainties
            venvShellHook
        ]))
    ];

    shellHook = ''
        python --version
    ''; 
}
