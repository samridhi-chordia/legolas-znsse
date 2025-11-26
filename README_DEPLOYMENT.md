# LEGOLAS ZnSSe Deployment Guide

## Overview

This guide covers deployment scenarios for the LEGOLAS ZnS(1-x)Se(x) DSSC optimization framework in different environments.

---

## Deployment Scenarios

### 1. Local Development/Testing Environment

**Use Case**: Research, development, testing, demonstrations

**Configuration**: Simulated mode with AFLOW fallback

```bash
# Quick deployment
cd /path/to/LEGOLAS_ZnSSe_AFLOW_RELEASE
pip install -r requirements.txt
python3 demo.py
```

**Mode Settings**:
```python
from znsse_interface import ZnSSeInterface

interface = ZnSSeInterface(
    mode='simulated',  # No hardware required
    egap_method='aflow_with_fallback'  # Try AFLOW, fallback to Vegard
)
```

**System Requirements**:
- Python 3.8+
- 150 MB RAM
- No special hardware
- Internet connection (optional, for AFLOW API)

---

### 2. Raspberry Pi + Hardware Deployment

**Use Case**: Real DSSC measurements in laboratory

**Hardware Requirements**:
- Raspberry Pi (any model with GPIO)
- MCP3008 ADC chip
- DSSC device
- Breadboard and wires
- Power supply (5V, 2.5A)

**Installation Steps**:

```bash
# 1. Enable SPI on Raspberry Pi
sudo raspi-config
# Navigate to: Interfacing Options → SPI → Enable → Reboot

# 2. Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-dev

# 3. Install Python packages
pip3 install -r requirements.txt
pip3 install RPi.GPIO spidev

# 4. Test hardware connection
python3 -c "import spidev; spi = spidev.SpiDev(); spi.open(0,0); print('SPI OK')"

# 5. Run with hardware
python3 demo.py
```

**Hardware Wiring**:

```
MCP3008 → Raspberry Pi
─────────────────────
VDD  → Pin 1  (3.3V)
VREF → Pin 1  (3.3V)
AGND → Pin 6  (GND)
DGND → Pin 6  (GND)
CLK  → Pin 23 (SCLK/GPIO 11)
DOUT → Pin 21 (MISO/GPIO 9)
DIN  → Pin 19 (MOSI/GPIO 10)
CS   → Pin 24 (CE0/GPIO 8)

DSSC+ → CH1 (channel 1)
DSSC- → AGND
```

**Mode Settings**:
```python
interface = ZnSSeInterface(
    mode='hardware_with_fallback',  # Try hardware, fallback to simulation
    egap_method='aflow_with_fallback'
)
```

---

### 3. Production Laboratory Deployment

**Use Case**: Continuous autonomous optimization experiments

**Architecture**:

```
┌─────────────────┐
│  Raspberry Pi   │
│  + MCP3008 ADC  │
│  + LEGOLAS Code │
└────────┬────────┘
         │
         │ USB/Network
         ▼
┌─────────────────┐
│  Central Server │
│  - Data Storage │
│  - Monitoring   │
│  - Backup       │
└─────────────────┘
```

**Deployment Steps**:

```bash
# On Raspberry Pi
git clone <repository>
cd LEGOLAS_ZnSSe_AFLOW_RELEASE
pip3 install -r requirements.txt
pip3 install RPi.GPIO spidev

# Setup as systemd service
sudo cp deployment/legolas.service /etc/systemd/system/
sudo systemctl enable legolas
sudo systemctl start legolas

# Monitor logs
sudo journalctl -u legolas -f
```

**Sample systemd service** (`deployment/legolas.service`):

```ini
[Unit]
Description=LEGOLAS ZnSSe Optimization Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/LEGOLAS_ZnSSe_AFLOW_RELEASE
ExecStart=/usr/bin/python3 /home/pi/LEGOLAS_ZnSSe_AFLOW_RELEASE/demo.py
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

---

### 4. Educational Workshop Deployment

**Use Case**: Classroom demonstrations, hands-on learning

**Setup**:

```bash
# 1. Clone to shared network location
mkdir /shared/LEGOLAS_Workshop
cp -r LEGOLAS_ZnSSe_AFLOW_RELEASE/* /shared/LEGOLAS_Workshop/

# 2. Create student environment
cd /shared/LEGOLAS_Workshop
chmod -R 755 .

# 3. Students access via network
# Each student runs:
python3 demo.py --output results_student<ID>.csv
```

**Configuration**: Simulated mode for all students

```python
# demo_workshop.py
interface = ZnSSeInterface(
    mode='simulated',  # Fast, no hardware required
    egap_method='vegard'  # Offline, deterministic
)

optimizer = GPOptimizer(interface, xi=0.01, random_state=None)
results = optimizer.optimize(n_iterations=5, n_initial=2)
```

---

## Configuration Files

### Environment Variables

Create `.env` file:

```bash
# Operation mode
LEGOLAS_MODE=hardware_with_fallback  # or 'simulated', 'hardware'
LEGOLAS_EGAP_METHOD=aflow_with_fallback  # or 'vegard', 'aflow'

# AFLOW API settings
AFLOW_API_URL=http://aflowlib.org/API/search/
AFLOW_TIMEOUT=10  # seconds

# Hardware settings
MCP3008_SPI_BUS=0
MCP3008_SPI_DEVICE=0
MCP3008_CHANNEL=1

# Results directory
RESULTS_DIR=./results

# Logging
LOG_LEVEL=INFO  # or DEBUG, WARNING, ERROR
LOG_FILE=legolas.log
```

### Load environment in Python:

```python
import os
from dotenv import load_dotenv

load_dotenv()

mode = os.getenv('LEGOLAS_MODE', 'simulated')
egap_method = os.getenv('LEGOLAS_EGAP_METHOD', 'vegard')

interface = ZnSSeInterface(mode=mode, egap_method=egap_method)
```

---

## Network Deployment

### Remote Access Setup

**SSH Access to Raspberry Pi**:

```bash
# Enable SSH
sudo raspi-config
# Interfacing Options → SSH → Enable

# Connect from laptop
ssh pi@raspberrypi.local

# Run optimization remotely
python3 /home/pi/LEGOLAS/demo.py
```

**File Transfer**:

```bash
# Upload code
scp -r LEGOLAS_ZnSSe_AFLOW_RELEASE pi@raspberrypi.local:/home/pi/

# Download results
scp pi@raspberrypi.local:/home/pi/LEGOLAS/results/* ./local_results/
```

**VNC for GUI** (optional):

```bash
# Enable VNC on Pi
sudo raspi-config
# Interfacing Options → VNC → Enable

# Connect from laptop using VNC Viewer
# Address: raspberrypi.local:5900
```

---

## Cloud Integration (Advanced)

### Upload Results to Cloud Storage

```python
import boto3
from datetime import datetime

def upload_results_to_s3(results_df, bucket='legolas-results'):
    """Upload optimization results to AWS S3"""
    s3 = boto3.client('s3')

    filename = f"optimization_{datetime.now().isoformat()}.csv"
    results_df.to_csv(f'/tmp/{filename}', index=False)

    s3.upload_file(
        f'/tmp/{filename}',
        bucket,
        filename
    )
    print(f"Results uploaded to s3://{bucket}/{filename}")

# After optimization
optimizer = GPOptimizer(interface)
results = optimizer.optimize(n_iterations=10, n_initial=4)
upload_results_to_s3(results)
```

---

## Monitoring & Logging

### Setup Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legolas.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('LEGOLAS')
logger.info("Starting optimization...")
```

### Monitor System Resources

```bash
# Install monitoring tools
pip3 install psutil

# Create monitoring script
cat > monitor.py << 'EOF'
import psutil
import time

while True:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    print(f"CPU: {cpu}%, RAM: {mem}%")
    time.sleep(5)
EOF

# Run in background
python3 monitor.py &
```

---

## Backup & Recovery

### Automated Backup

```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR=/backup/legolas
DATE=$(date +%Y%m%d_%H%M%S)

# Backup code and results
tar -czf $BACKUP_DIR/legolas_$DATE.tar.gz \
    LEGOLAS_ZnSSe_AFLOW_RELEASE/ \
    --exclude='*.pyc' \
    --exclude='__pycache__'

echo "Backup completed: legolas_$DATE.tar.gz"
EOF

chmod +x backup.sh

# Schedule daily backups
crontab -e
# Add: 0 2 * * * /home/pi/backup.sh
```

### Recovery

```bash
# Restore from backup
cd /home/pi
tar -xzf /backup/legolas/legolas_20250116_020000.tar.gz
cd LEGOLAS_ZnSSe_AFLOW_RELEASE
pip3 install -r requirements.txt
```

---

## Security Considerations

### File Permissions

```bash
# Set appropriate permissions
chmod 755 *.py
chmod 644 data/*.csv
chmod 700 results/

# Restrict access to configuration
chmod 600 .env
```

### Network Security

```bash
# Firewall rules (if running server)
sudo ufw allow ssh
sudo ufw allow 5000/tcp  # If running web interface
sudo ufw enable
```

---

## Performance Optimization

### Parallel Processing

```python
from multiprocessing import Pool

def optimize_parallel(compositions):
    """Run optimization for multiple compositions in parallel"""
    with Pool(processes=4) as pool:
        results = pool.map(interface.measure_voltage, compositions)
    return results

compositions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = optimize_parallel(compositions)
```

### Memory Management

```python
# For large-scale experiments
import gc

for iteration in range(1000):
    result = interface.measure_voltage(x_Se)
    # Process result

    if iteration % 100 == 0:
        gc.collect()  # Free memory
```

---

## Troubleshooting

### Common Deployment Issues

**1. Permission Denied (Raspberry Pi)**

```bash
# Add user to necessary groups
sudo usermod -a -G spi,gpio,i2c $USER
# Log out and back in
```

**2. Network Connection Issues**

```bash
# Test AFLOW connectivity
curl -I http://aflowlib.org/API/search/
# Should return: HTTP/1.1 200 OK

# If blocked, use proxy
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

**3. Out of Memory**

```bash
# Increase swap space (Raspberry Pi)
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set: CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

**4. SPI Not Working**

```bash
# Verify SPI enabled
lsmod | grep spi
# Should show: spi_bcm2835

# Check device
ls -l /dev/spidev*
# Should show: /dev/spidev0.0 and /dev/spidev0.1
```

---

## Scaling to Multiple Devices

### Distributed Setup

```python
# coordinator.py
from multiprocessing import Process
import zmq

def worker(worker_id):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://coordinator:5555")

    interface = ZnSSeInterface(mode='hardware', device_id=worker_id)

    while True:
        socket.send_json({'status': 'ready', 'worker': worker_id})
        task = socket.recv_json()

        if task['type'] == 'measure':
            result = interface.measure_voltage(task['x_Se'])
            socket.send_json(result)

# Launch workers
processes = []
for i in range(4):  # 4 Raspberry Pi devices
    p = Process(target=worker, args=(i,))
    p.start()
    processes.append(p)
```

---

## Maintenance Schedule

### Daily Tasks

- Check system logs: `journalctl -u legolas -since today`
- Verify disk space: `df -h`
- Monitor results: `ls -lh results/`

### Weekly Tasks

- Backup results: `./backup.sh`
- Update packages: `pip3 install --upgrade -r requirements.txt`
- Clean temporary files: `rm -rf __pycache__`

### Monthly Tasks

- System updates: `sudo apt-get update && sudo apt-get upgrade`
- Review optimization performance
- Archive old results

---

## Support & Documentation

- **Installation Guide**: See README_INSTALLATION.md
- **Functional Overview**: See README.md
- **Contact**: schordi2@jhu.edu | samridhichordia@gmail.com

---

**Last Updated**: November 2025
**Version**: 1.0
