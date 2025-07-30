#!/usr/bin/env python3
"""
SBOM (Software Bill of Materials) Generation Script

Generates comprehensive SBOMs in multiple formats for supply chain security.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def run_command(cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    try:
        return subprocess.run(
            cmd, 
            capture_output=capture_output,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def generate_cyclonedx_sbom(output_dir: Path) -> Path:
    """Generate SBOM using CycloneDX."""
    output_file = output_dir / "sbom-cyclonedx.json"
    
    print("Generating CycloneDX SBOM...")
    run_command([
        "cyclonedx-py",
        "--output-format", "json",
        "--output-file", str(output_file),
        "."
    ])
    
    return output_file


def generate_syft_sbom(output_dir: Path) -> Path:
    """Generate SBOM using Syft."""
    output_file = output_dir / "sbom-syft.spdx.json"
    
    print("Generating Syft SBOM...")
    run_command([
        "syft",
        "packages",
        "dir:.",
        "-o", f"spdx-json={output_file}"
    ])
    
    return output_file


def validate_sbom(sbom_file: Path) -> bool:
    """Validate SBOM format and content."""
    print(f"Validating SBOM: {sbom_file}")
    
    try:
        with open(sbom_file) as f:
            data = json.load(f)
            
        # Basic validation
        required_fields = ["components", "metadata"]
        if sbom_file.suffix == ".json" and "bomFormat" in data:
            # CycloneDX format
            required_fields = ["bomFormat", "specVersion", "components"]
        elif "spdxVersion" in data:
            # SPDX format
            required_fields = ["SPDXID", "spdxVersion", "packages"]
            
        for field in required_fields:
            if field not in data:
                print(f"Missing required field: {field}")
                return False
                
        print("SBOM validation passed")
        return True
        
    except Exception as e:
        print(f"SBOM validation failed: {e}")
        return False


def sign_sbom(sbom_file: Path) -> Optional[Path]:
    """Sign SBOM using cosign."""
    try:
        signature_file = sbom_file.with_suffix(sbom_file.suffix + ".sig")
        
        print(f"Signing SBOM: {sbom_file}")
        run_command([
            "cosign", "sign-blob",
            "--output-signature", str(signature_file),
            str(sbom_file)
        ])
        
        return signature_file
        
    except Exception as e:
        print(f"SBOM signing failed (optional): {e}")
        return None


def generate_vulnerability_report(sbom_file: Path, output_dir: Path) -> Path:
    """Generate vulnerability report from SBOM."""
    vuln_file = output_dir / f"vulnerabilities-{sbom_file.stem}.json"
    
    try:
        print("Generating vulnerability report...")
        run_command([
            "grype",
            f"sbom:{sbom_file}",
            "-o", "json",
            "--file", str(vuln_file)
        ])
        
        return vuln_file
        
    except Exception as e:
        print(f"Vulnerability scanning failed (optional): {e}")
        return None


def main():
    """Main SBOM generation workflow."""
    print("PWMK SBOM Generation")
    print("===================")
    
    # Setup output directory
    output_dir = Path("dist")
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration
    config_file = Path("sbom-config.json")
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"Using configuration: {config_file}")
    else:
        config = {}
        print("Using default configuration")
    
    # Generate SBOMs
    sbom_files = []
    
    try:
        cyclonedx_sbom = generate_cyclonedx_sbom(output_dir)
        sbom_files.append(cyclonedx_sbom)
    except Exception as e:
        print(f"CycloneDX generation failed: {e}")
    
    try:
        syft_sbom = generate_syft_sbom(output_dir)
        sbom_files.append(syft_sbom)
    except Exception as e:
        print(f"Syft generation failed: {e}")
    
    if not sbom_files:
        print("No SBOMs generated successfully")
        sys.exit(1)
    
    # Validate and process SBOMs
    for sbom_file in sbom_files:
        if validate_sbom(sbom_file):
            # Sign SBOM if configured
            if config.get("generation", {}).get("signing", {}).get("enabled"):
                sign_sbom(sbom_file)
            
            # Generate vulnerability report
            if config.get("components", {}).get("vulnerability_scanning"):
                generate_vulnerability_report(sbom_file, output_dir)
    
    print(f"\nSBOM generation complete. Files saved to: {output_dir}")
    print("Generated files:")
    for file in output_dir.glob("*"):
        print(f"  - {file}")


if __name__ == "__main__":
    main()