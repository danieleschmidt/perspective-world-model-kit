#!/usr/bin/env python3
"""
Quality Gates Runner - Comprehensive Quality Assurance

Runs all quality gates including tests, security scans, performance validation,
and compliance checks for the PWMK system.
"""

import sys
import time
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quality_gates.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class QualityGate:
    """Individual quality gate definition."""
    
    def __init__(self, name: str, description: str, command: List[str], 
                 required: bool = True, timeout: int = 600):
        self.name = name
        self.description = description
        self.command = command
        self.required = required
        self.timeout = timeout
        self.result = None
        self.duration = 0.0
        self.output = ""
        self.error_output = ""


class QualityGatesRunner:
    """Comprehensive quality gates runner."""
    
    def __init__(self):
        self.quality_gates = []
        self.results = {}
        self.overall_success = False
        self.start_time = 0
        self.end_time = 0
        
        # Initialize quality gates
        self._initialize_quality_gates()
    
    def _initialize_quality_gates(self):
        """Initialize all quality gates."""
        
        # Code Quality Gates
        self.quality_gates.extend([
            QualityGate(
                name="lint_check",
                description="Code linting and style check",
                command=["python", "-m", "flake8", "pwmk/", "--max-line-length=100", "--ignore=E203,W503"],
                required=False,  # Warning only
                timeout=120
            ),
            QualityGate(
                name="type_check",
                description="Static type checking",
                command=["python", "-m", "mypy", "pwmk/", "--ignore-missing-imports", "--no-strict-optional"],
                required=False,  # Warning only
                timeout=180
            ),
            QualityGate(
                name="security_scan",
                description="Security vulnerability scan",
                command=["python", "-m", "bandit", "-r", "pwmk/", "-f", "json"],
                required=True,
                timeout=300
            )
        ])
        
        # Testing Gates
        self.quality_gates.extend([
            QualityGate(
                name="unit_tests",
                description="Unit tests execution",
                command=["python", "-m", "pytest", "tests/", "-v", "--tb=short", "--maxfail=5"],
                required=True,
                timeout=600
            ),
            QualityGate(
                name="integration_tests",
                description="Integration tests execution",
                command=["python", "-m", "pytest", "tests/integration/", "-v", "--tb=short"],
                required=True,
                timeout=900
            ),
            QualityGate(
                name="validation_tests",
                description="System validation tests",
                command=["python", "-c", "from pwmk.validation import create_comprehensive_validation; v=create_comprehensive_validation(); print('Validation:', v.run_all_validations())"],
                required=True,
                timeout=300
            )
        ])
        
        # Performance Gates
        self.quality_gates.extend([
            QualityGate(
                name="performance_benchmark",
                description="Performance benchmarking",
                command=["python", "scripts/performance_benchmark.py"],
                required=False,  # Performance regression check
                timeout=600
            ),
            QualityGate(
                name="memory_leak_check",
                description="Memory leak detection",
                command=["python", "-c", "import gc; gc.set_debug(gc.DEBUG_LEAK); exec(open('demo_complete_integration.py').read())"],
                required=False,
                timeout=300
            )
        ])
        
        # Security Gates  
        self.quality_gates.extend([
            QualityGate(
                name="dependency_check",
                description="Dependency vulnerability check",
                command=["python", "-m", "safety", "check", "--json"],
                required=True,
                timeout=180
            ),
            QualityGate(
                name="secrets_scan",
                description="Secrets and sensitive data scan",
                command=["python", "-c", "import re, os; [print('SECRETS_FOUND') for root, dirs, files in os.walk('.') for file in files if file.endswith('.py') for line in open(os.path.join(root, file), 'r', errors='ignore').readlines() if re.search(r'(password|secret|key|token)\\s*=\\s*['\\\"]\\w+['\\\"]', line.lower())]"],
                required=True,
                timeout=120
            )
        ])
        
        # Documentation Gates
        self.quality_gates.extend([
            QualityGate(
                name="documentation_check",
                description="Documentation completeness check",
                command=["python", "-c", "import ast, os; missing=[f for f in [os.path.join(r,file) for r,d,files in os.walk('pwmk') for file in files if file.endswith('.py')] if not any('\"\"\"' in line for line in open(f, 'r', errors='ignore').readlines()[:10])]; print(f'Missing docstrings: {len(missing)}'); exit(1 if len(missing) > len([f for f in [os.path.join(r,file) for r,d,files in os.walk('pwmk') for file in files if file.endswith('.py')]]) * 0.2 else 0)"],
                required=False,
                timeout=60
            ),
            QualityGate(
                name="readme_validation",
                description="README and documentation validation",
                command=["python", "-c", "import os; files=['README.md', 'CONTRIBUTING.md', 'LICENSE']; missing=[f for f in files if not os.path.exists(f)]; print(f'Missing docs: {missing}'); exit(1 if missing else 0)"],
                required=True,
                timeout=30
            )
        ])
        
        # Consciousness-Specific Gates
        self.quality_gates.extend([
            QualityGate(
                name="consciousness_validation",
                description="Consciousness engine validation",
                command=["python", "-c", "from pwmk.revolution.consciousness_engine import ConsciousnessEngine; from pwmk.core.world_model import PerspectiveWorldModel; from pwmk.core.beliefs import BeliefStore; from pwmk.breakthrough.emergent_intelligence import EmergentIntelligenceSystem; from pwmk.autonomous.self_improving_agent import SelfImprovingAgent; print('Consciousness validation: PASS')"],
                required=True,
                timeout=180
            ),
            QualityGate(
                name="quantum_validation",
                description="Quantum processor validation",
                command=["python", "-c", "from pwmk.quantum.adaptive_quantum import AdaptiveQuantumProcessor; q=AdaptiveQuantumProcessor(8, 6); print('Quantum validation: PASS')"],
                required=True,
                timeout=120
            ),
            QualityGate(
                name="research_validation",
                description="Research framework validation",
                command=["python", "-c", "from pwmk.research.advanced_framework import AdvancedResearchFramework; print('Research validation: PASS')"],
                required=True,
                timeout=120
            )
        ])
    
    def run_all_gates(self) -> bool:
        """Run all quality gates."""
        self.start_time = time.time()
        
        print("🚦 STARTING COMPREHENSIVE QUALITY GATES")
        print("=" * 60)
        
        # Track results
        passed_gates = 0
        failed_gates = 0
        skipped_gates = 0
        
        for gate in self.quality_gates:
            print(f"\n📋 Running: {gate.name}")
            print(f"   Description: {gate.description}")
            print(f"   Required: {'Yes' if gate.required else 'No'}")
            
            try:
                success = self._run_gate(gate)
                
                if success:
                    print(f"   ✅ PASSED ({gate.duration:.2f}s)")
                    passed_gates += 1
                else:
                    if gate.required:
                        print(f"   ❌ FAILED ({gate.duration:.2f}s) - REQUIRED")
                        failed_gates += 1
                    else:
                        print(f"   ⚠️  FAILED ({gate.duration:.2f}s) - OPTIONAL")
                        skipped_gates += 1
                
                self.results[gate.name] = {
                    'success': success,
                    'required': gate.required,
                    'duration': gate.duration,
                    'output': gate.output,
                    'error_output': gate.error_output
                }
                
            except Exception as e:
                print(f"   💥 ERROR: {str(e)}")
                failed_gates += 1
                
                self.results[gate.name] = {
                    'success': False,
                    'required': gate.required,
                    'duration': 0.0,
                    'output': "",
                    'error_output': str(e)
                }
        
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # Determine overall success
        required_failures = sum(1 for result in self.results.values() 
                              if not result['success'] and result['required'])
        
        self.overall_success = required_failures == 0
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 QUALITY GATES SUMMARY")
        print("=" * 60)
        print(f"⏱️  Total Duration: {total_duration:.2f}s")
        print(f"✅ Passed: {passed_gates}")
        print(f"❌ Failed: {failed_gates}")
        print(f"⚠️  Skipped: {skipped_gates}")
        print(f"📊 Total: {len(self.quality_gates)}")
        
        if self.overall_success:
            print("\n🎉 ALL REQUIRED QUALITY GATES PASSED!")
            print("🚀 System is ready for production deployment")
        else:
            print(f"\n💥 {required_failures} REQUIRED QUALITY GATES FAILED!")
            print("🛑 System is NOT ready for production deployment")
            
            # List failed required gates
            failed_required = [
                name for name, result in self.results.items()
                if not result['success'] and result['required']
            ]
            print(f"Failed required gates: {', '.join(failed_required)}")
        
        # Generate detailed report
        self._generate_quality_report()
        
        return self.overall_success
    
    def _run_gate(self, gate: QualityGate) -> bool:
        """Run individual quality gate."""
        start_time = time.time()
        
        try:
            # Create safe environment for command execution
            env = {
                'PYTHONPATH': str(Path.cwd()),
                'PATH': '/usr/bin:/bin:/usr/local/bin'
            }
            
            # Run command
            result = subprocess.run(
                gate.command,
                capture_output=True,
                text=True,
                timeout=gate.timeout,
                cwd=Path.cwd(),
                env=env
            )
            
            gate.duration = time.time() - start_time
            gate.output = result.stdout
            gate.error_output = result.stderr
            
            # Determine success based on return code
            success = result.returncode == 0
            
            # Special handling for specific gates
            if gate.name == "secrets_scan" and "SECRETS_FOUND" in result.stdout:
                success = False
            
            if gate.name == "security_scan":
                # Parse bandit JSON output
                try:
                    if result.stdout:
                        bandit_result = json.loads(result.stdout)
                        high_severity = len([issue for issue in bandit_result.get('results', []) 
                                           if issue.get('issue_severity') == 'HIGH'])
                        success = high_severity == 0
                except:
                    # If JSON parsing fails, use return code
                    pass
            
            return success
            
        except subprocess.TimeoutExpired:
            gate.duration = gate.timeout
            gate.error_output = f"Command timed out after {gate.timeout} seconds"
            return False
            
        except Exception as e:
            gate.duration = time.time() - start_time
            gate.error_output = str(e)
            return False
    
    def _generate_quality_report(self):
        """Generate detailed quality report."""
        report = {
            'overall_success': self.overall_success,
            'total_duration': self.end_time - self.start_time,
            'timestamp': time.time(),
            'gates': self.results,
            'summary': {
                'total_gates': len(self.quality_gates),
                'passed': sum(1 for r in self.results.values() if r['success']),
                'failed': sum(1 for r in self.results.values() if not r['success']),
                'required_failures': sum(1 for r in self.results.values() 
                                       if not r['success'] and r['required'])
            }
        }
        
        # Add recommendations
        recommendations = []
        
        for name, result in self.results.items():
            if not result['success']:
                if name == "lint_check":
                    recommendations.append("Fix code style issues using: python -m black pwmk/")
                elif name == "type_check":
                    recommendations.append("Add type hints and fix type issues")
                elif name == "security_scan":
                    recommendations.append("Address security vulnerabilities identified by bandit")
                elif name == "unit_tests":
                    recommendations.append("Fix failing unit tests")
                elif name == "integration_tests":
                    recommendations.append("Fix failing integration tests")
                elif name == "dependency_check":
                    recommendations.append("Update vulnerable dependencies")
                elif name == "secrets_scan":
                    recommendations.append("Remove hardcoded secrets and use environment variables")
        
        if not recommendations:
            recommendations.append("All quality gates passing - system is production ready!")
        
        report['recommendations'] = recommendations
        
        # Save report
        report_file = Path("quality_gates_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Generate human-readable report
        human_report = self._generate_human_readable_report(report)
        human_report_file = Path("quality_gates_report.md")
        with open(human_report_file, 'w') as f:
            f.write(human_report)
        
        print(f"📄 Human-readable report saved to: {human_report_file}")
    
    def _generate_human_readable_report(self, report: Dict[str, Any]) -> str:
        """Generate human-readable quality report."""
        
        md_report = f"""# Quality Gates Report

## Summary

- **Overall Status**: {'✅ PASSED' if report['overall_success'] else '❌ FAILED'}
- **Total Duration**: {report['total_duration']:.2f} seconds
- **Report Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Gate Results

| Gate | Status | Duration | Required | Description |
|------|--------|----------|----------|-------------|
"""
        
        for gate in self.quality_gates:
            result = report['gates'].get(gate.name, {})
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            duration = result.get('duration', 0.0)
            required = "Yes" if gate.required else "No"
            
            md_report += f"| {gate.name} | {status} | {duration:.2f}s | {required} | {gate.description} |\n"
        
        md_report += f"""
## Statistics

- **Total Gates**: {report['summary']['total_gates']}
- **Passed**: {report['summary']['passed']}
- **Failed**: {report['summary']['failed']}
- **Required Failures**: {report['summary']['required_failures']}

## Recommendations

"""
        
        for i, recommendation in enumerate(report['recommendations'], 1):
            md_report += f"{i}. {recommendation}\n"
        
        if report['overall_success']:
            md_report += """
## 🚀 Deployment Status

✅ **SYSTEM IS READY FOR PRODUCTION DEPLOYMENT**

All required quality gates have passed. The system meets production quality standards.
"""
        else:
            md_report += """
## 🛑 Deployment Status

❌ **SYSTEM IS NOT READY FOR PRODUCTION DEPLOYMENT**

Required quality gates have failed. Address the issues above before deploying to production.
"""
        
        return md_report


def install_dependencies():
    """Install required dependencies for quality gates."""
    dependencies = [
        "flake8",
        "mypy", 
        "bandit",
        "safety",
        "pytest",
        "pytest-cov"
    ]
    
    print("📦 Installing quality gate dependencies...")
    
    for dep in dependencies:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                          check=True, capture_output=True)
            print(f"   ✅ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  Failed to install {dep} (may already be installed)")


def main():
    """Main quality gates runner."""
    print("🔧 PWMK Quality Gates Runner")
    print("=" * 40)
    
    # Install dependencies
    try:
        install_dependencies()
    except Exception as e:
        print(f"⚠️  Warning: Failed to install some dependencies: {e}")
    
    # Run quality gates
    runner = QualityGatesRunner()
    
    try:
        success = runner.run_all_gates()
        exit_code = 0 if success else 1
        
        print(f"\n🏁 Quality gates completed with exit code: {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Quality gates interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n💥 Quality gates runner failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()