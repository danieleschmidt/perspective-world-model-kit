# Disaster Recovery Plan

## Overview

This document outlines the disaster recovery procedures for the Perspective World Model Kit (PWMK) project, including backup strategies, recovery procedures, and business continuity planning.

## Risk Assessment

### Critical Assets

1. **Source Code**: Git repository and development history
2. **Training Data**: Research datasets and experiment results
3. **Trained Models**: Pre-trained world models and checkpoints
4. **Documentation**: Technical documentation and research papers
5. **Configuration**: Deployment and infrastructure configurations
6. **Secrets**: API keys, certificates, and access tokens

### Threat Categories

| Threat Type | Probability | Impact | Risk Level |
|-------------|-------------|---------|------------|
| Hardware Failure | High | Medium | High |
| Data Corruption | Medium | High | High |
| Cyber Attack | Medium | High | High |
| Human Error | High | Medium | High |
| Natural Disaster | Low | High | Medium |
| Cloud Provider Outage | Medium | Medium | Medium |

## Backup Strategy

### Repository Backup

#### Git Repository Protection
```bash
# Multiple repository mirrors
git remote add backup-1 git@github.com:backup-org/pwmk-backup.git
git remote add backup-2 git@gitlab.com:backup-org/pwmk-backup.git
git remote add backup-3 git@bitbucket.org:backup-org/pwmk-backup.git

# Automated daily backup script
#!/bin/bash
# scripts/backup_repository.sh

set -euo pipefail

BACKUP_DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/pwmk-$BACKUP_DATE"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup main repository
git bundle create "$BACKUP_DIR/pwmk-main.bundle" --all

# Backup all branches and tags
git push --all backup-1
git push --tags backup-1
git push --all backup-2  
git push --tags backup-2

# Create archive of working directory
tar -czf "$BACKUP_DIR/pwmk-workspace.tar.gz" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='node_modules' \
    .

# Upload to cloud storage
aws s3 cp "$BACKUP_DIR/" s3://pwmk-backups/repository/ --recursive

echo "Repository backup completed: $BACKUP_DIR"
```

### Data Backup

#### Research Data Protection
```bash
# Automated data backup script
#!/bin/bash
# scripts/backup_data.sh

BACKUP_DATE=$(date +%Y%m%d)
DATA_DIRS=("data" "experiments" "models" "checkpoints")

for dir in "${DATA_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        # Create incremental backup
        rsync -av --backup --backup-dir="backups/$BACKUP_DATE" \
            "$dir/" "s3://pwmk-data-backups/$dir/"
        
        # Create compressed archive
        tar -czf "backups/$dir-$BACKUP_DATE.tar.gz" "$dir/"
        
        # Upload to multiple cloud providers
        aws s3 cp "backups/$dir-$BACKUP_DATE.tar.gz" \
            s3://pwmk-primary-backups/
        
        gsutil cp "backups/$dir-$BACKUP_DATE.tar.gz" \
            gs://pwmk-secondary-backups/
    fi
done
```

#### Model Checkpoint Backup
```python
# pwmk/utils/model_backup.py
import boto3
from datetime import datetime
import torch
import json
import hashlib

class ModelBackupManager:
    """Automated model checkpoint backup system."""
    
    def __init__(self, 
                 primary_bucket: str = "pwmk-model-backups",
                 secondary_bucket: str = "pwmk-model-backups-secondary"):
        self.primary_bucket = primary_bucket
        self.secondary_bucket = secondary_bucket
        self.s3_client = boto3.client('s3')
    
    def backup_checkpoint(self, 
                         model: torch.nn.Module,
                         checkpoint_name: str,
                         metadata: dict = None):
        """Backup model checkpoint with metadata."""
        
        timestamp = datetime.utcnow().isoformat()
        backup_path = f"checkpoints/{timestamp}/{checkpoint_name}"
        
        # Save model state
        model_state = {
            'state_dict': model.state_dict(),
            'model_config': model.config,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Calculate checksum
        state_str = str(model_state)
        checksum = hashlib.sha256(state_str.encode()).hexdigest()
        model_state['checksum'] = checksum
        
        # Save to temporary file
        temp_path = f"/tmp/{checkpoint_name}_{timestamp}.pt"
        torch.save(model_state, temp_path)
        
        # Upload to primary storage
        self._upload_file(temp_path, backup_path, self.primary_bucket)
        
        # Upload to secondary storage (async)
        self._upload_file(temp_path, backup_path, self.secondary_bucket)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return backup_path
    
    def _upload_file(self, local_path: str, remote_path: str, bucket: str):
        """Upload file to S3 bucket."""
        try:
            self.s3_client.upload_file(local_path, bucket, remote_path)
            print(f"Backup uploaded: s3://{bucket}/{remote_path}")
        except Exception as e:
            print(f"Backup failed for {bucket}: {e}")
```

## Recovery Procedures

### Source Code Recovery

#### Git Repository Recovery
```bash
# Recovery from bundle backup
git clone pwmk-main.bundle pwmk-recovered
cd pwmk-recovered

# Restore remote repositories
git remote add origin git@github.com:your-org/pwmk.git
git push --all origin
git push --tags origin

# Verify integrity
git fsck --full
```

#### Working Directory Recovery
```bash
# Extract workspace backup
tar -xzf pwmk-workspace.tar.gz -C pwmk-recovered/

# Restore virtual environment
cd pwmk-recovered
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test]"

# Verify functionality
make test
```

### Data Recovery

#### Research Data Recovery
```bash
# Download from primary backup
aws s3 sync s3://pwmk-data-backups/data/ ./data/
aws s3 sync s3://pwmk-data-backups/experiments/ ./experiments/

# Verify data integrity
find data/ -name "*.pkl" -exec python -c "import pickle; pickle.load(open('{}', 'rb'))" \;
find experiments/ -name "*.json" -exec python -c "import json; json.load(open('{}'))" \;
```

#### Model Recovery
```python
# pwmk/utils/model_recovery.py
import torch
import boto3
from typing import Optional

class ModelRecoveryManager:
    """Model checkpoint recovery system."""
    
    def __init__(self, backup_bucket: str = "pwmk-model-backups"):
        self.backup_bucket = backup_bucket
        self.s3_client = boto3.client('s3')
    
    def list_available_checkpoints(self) -> list:
        """List all available model checkpoints."""
        response = self.s3_client.list_objects_v2(
            Bucket=self.backup_bucket,
            Prefix="checkpoints/"
        )
        
        checkpoints = []
        for obj in response.get('Contents', []):
            checkpoints.append({
                'key': obj['Key'],
                'size': obj['Size'],
                'last_modified': obj['LastModified']
            })
        
        return sorted(checkpoints, key=lambda x: x['last_modified'], reverse=True)
    
    def recover_checkpoint(self, 
                          checkpoint_path: str,
                          local_path: str = None) -> dict:
        """Recover model checkpoint from backup."""
        
        if local_path is None:
            local_path = f"/tmp/{checkpoint_path.split('/')[-1]}"
        
        # Download checkpoint
        self.s3_client.download_file(
            self.backup_bucket,
            checkpoint_path,
            local_path
        )
        
        # Load and verify checkpoint
        checkpoint = torch.load(local_path, map_location='cpu')
        
        # Verify checksum
        expected_checksum = checkpoint.pop('checksum')
        actual_checksum = hashlib.sha256(str(checkpoint).encode()).hexdigest()
        
        if expected_checksum != actual_checksum:
            raise ValueError("Checkpoint integrity check failed")
        
        return checkpoint

# Usage example
recovery_manager = ModelRecoveryManager()
available_checkpoints = recovery_manager.list_available_checkpoints()
latest_checkpoint = recovery_manager.recover_checkpoint(
    available_checkpoints[0]['key']
)
```

## Business Continuity

### Service Restoration Priority

1. **Critical (< 1 hour)**
   - Git repository access
   - Development environment setup
   - Core documentation access

2. **High (< 4 hours)**
   - Training data restoration
   - Model checkpoint recovery
   - CI/CD pipeline restoration

3. **Medium (< 24 hours)**
   - Monitoring and observability
   - Development tools and integrations
   - Performance benchmarks

4. **Low (< 72 hours)**
   - Historical experiment data
   - Advanced analytics dashboards
   - Non-critical documentation

### Communication Plan

#### Incident Response Team

| Role | Primary | Secondary | Contact |
|------|---------|-----------|---------|
| Incident Commander | John Doe | Jane Smith | +1-555-0101 |
| Technical Lead | Alice Johnson | Bob Wilson | +1-555-0102 |
| Infrastructure Lead | Carol Brown | David Lee | +1-555-0103 |
| Communications Lead | Eve Garcia | Frank Miller | +1-555-0104 |

#### Communication Channels

1. **Internal Communication**
   - Slack: #incident-response
   - Email: incident-team@pwmk.ai
   - Phone bridge: +1-555-BRIDGE

2. **External Communication**
   - Status page: https://status.pwmk.ai
   - Twitter: @PWMK_Status
   - Email: users@pwmk.ai

#### Communication Templates

```markdown
# Initial Incident Notification
Subject: [INCIDENT] PWMK Service Disruption - {{ severity }}

We are experiencing a {{ type }} incident affecting {{ affected_services }}.

**Impact**: {{ impact_description }}
**Started**: {{ start_time }}
**ETA**: {{ estimated_resolution }}

We are actively investigating and will provide updates every 30 minutes.

Status page: https://status.pwmk.ai/incident/{{ incident_id }}
```

```markdown
# Resolution Notification
Subject: [RESOLVED] PWMK Service Disruption

The incident affecting {{ affected_services }} has been resolved.

**Duration**: {{ total_duration }}
**Root Cause**: {{ root_cause_summary }}
**Resolution**: {{ resolution_summary }}

A detailed post-mortem will be published within 48 hours.
```

### Recovery Testing

#### Disaster Recovery Drills

```bash
# Quarterly DR drill script
#!/bin/bash
# scripts/dr_drill.sh

echo "Starting Disaster Recovery Drill: $(date)"

# Test 1: Repository Recovery
echo "Testing repository recovery..."
./scripts/test_repository_recovery.sh

# Test 2: Data Recovery
echo "Testing data recovery..."
./scripts/test_data_recovery.sh

# Test 3: Model Recovery
echo "Testing model recovery..."
python scripts/test_model_recovery.py

# Test 4: Service Restoration
echo "Testing service restoration..."
./scripts/test_service_restoration.sh

# Generate drill report
python scripts/generate_dr_report.py

echo "Disaster Recovery Drill completed: $(date)"
```

#### Recovery Validation

```python
# scripts/validate_recovery.py
import subprocess
import json
import sys
from datetime import datetime

class RecoveryValidator:
    """Validate disaster recovery procedures."""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.utcnow()
    
    def validate_repository_recovery(self) -> bool:
        """Validate git repository recovery."""
        try:
            # Test repository clone
            result = subprocess.run([
                'git', 'clone', '--quiet', 
                'pwmk-test.bundle', 
                'recovery-test'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Verify commit history
                result = subprocess.run([
                    'git', '-C', 'recovery-test', 
                    'log', '--oneline', '-n', '10'
                ], capture_output=True, text=True)
                
                return len(result.stdout.strip().split('\n')) >= 10
            
            return False
        except Exception as e:
            print(f"Repository recovery validation failed: {e}")
            return False
    
    def validate_data_recovery(self) -> bool:
        """Validate data recovery procedures."""
        try:
            import pickle
            import json
            
            # Test data file integrity
            data_files = [
                'data/train_data.pkl',
                'experiments/config.json',
                'models/checkpoint.pt'
            ]
            
            for file_path in data_files:
                if not os.path.exists(file_path):
                    return False
                
                # Basic integrity check
                if file_path.endswith('.pkl'):
                    with open(file_path, 'rb') as f:
                        pickle.load(f)
                elif file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        json.load(f)
            
            return True
        except Exception as e:
            print(f"Data recovery validation failed: {e}")
            return False
    
    def validate_service_restoration(self) -> bool:
        """Validate service restoration."""
        try:
            # Test application startup
            result = subprocess.run([
                'python', '-c', 'import pwmk; print("OK")'
            ], capture_output=True, text=True, timeout=30)
            
            return result.returncode == 0 and 'OK' in result.stdout
        except Exception as e:
            print(f"Service restoration validation failed: {e}")
            return False
    
    def run_all_validations(self) -> dict:
        """Run all recovery validations."""
        validations = {
            'repository_recovery': self.validate_repository_recovery,
            'data_recovery': self.validate_data_recovery,
            'service_restoration': self.validate_service_restoration
        }
        
        results = {}
        for name, validation_func in validations.items():
            try:
                results[name] = {
                    'status': 'PASS' if validation_func() else 'FAIL',
                    'timestamp': datetime.utcnow().isoformat()
                }
            except Exception as e:
                results[name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        # Overall status
        overall_status = 'PASS' if all(
            r['status'] == 'PASS' for r in results.values()
        ) else 'FAIL'
        
        return {
            'overall_status': overall_status,
            'duration_seconds': (datetime.utcnow() - self.start_time).total_seconds(),
            'validations': results,
            'timestamp': datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    validator = RecoveryValidator()
    results = validator.run_all_validations()
    
    print(json.dumps(results, indent=2))
    
    if results['overall_status'] != 'PASS':
        sys.exit(1)
```

## Monitoring and Alerting

### Backup Monitoring

```python
# scripts/monitor_backups.py
from datetime import datetime, timedelta
import boto3
import json

class BackupMonitor:
    """Monitor backup health and completeness."""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.expected_backups = [
            'repository/pwmk-main.bundle',
            'data/research_data.tar.gz',
            'models/latest_checkpoint.pt'
        ]
    
    def check_backup_freshness(self, bucket: str, max_age_hours: int = 24) -> dict:
        """Check if backups are recent enough."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        results = {}
        
        for backup in self.expected_backups:
            try:
                response = self.s3_client.head_object(Bucket=bucket, Key=backup)
                last_modified = response['LastModified'].replace(tzinfo=None)
                
                results[backup] = {
                    'exists': True,
                    'last_modified': last_modified.isoformat(),
                    'is_fresh': last_modified > cutoff_time,
                    'age_hours': (datetime.utcnow() - last_modified).total_seconds() / 3600
                }
            except Exception as e:
                results[backup] = {
                    'exists': False,
                    'error': str(e),
                    'is_fresh': False
                }
        
        return results
    
    def generate_backup_report(self) -> dict:
        """Generate comprehensive backup status report."""
        buckets = ['pwmk-primary-backups', 'pwmk-secondary-backups']
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'buckets': {}
        }
        
        for bucket in buckets:
            report['buckets'][bucket] = self.check_backup_freshness(bucket)
        
        # Overall health
        all_fresh = all(
            backup['is_fresh'] 
            for bucket_data in report['buckets'].values()
            for backup in bucket_data.values()
        )
        
        report['overall_health'] = 'HEALTHY' if all_fresh else 'UNHEALTHY'
        
        return report

# Prometheus metrics for backup monitoring
from prometheus_client import Gauge, Counter

backup_age_hours = Gauge(
    'pwmk_backup_age_hours',
    'Age of backup files in hours',
    ['backup_type', 'bucket']
)

backup_check_failures = Counter(
    'pwmk_backup_check_failures_total',
    'Total backup check failures',
    ['backup_type', 'bucket']
)
```

This comprehensive disaster recovery plan ensures PWMK can recover quickly from various failure scenarios while maintaining data integrity and minimizing downtime.