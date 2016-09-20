# mppTracker
python max power point tracker for solar cells (hopefully robust enough for perovskites)

## Usage
```
usage: mppTracker.py [-h] address t_dwell t_total

Max power point tracker for solar cells using a Keityhley 2400 sourcemeter
(hopefully robust enough for perovskites)

positional arguments:
  address     VISA resource name for sourcemeter
  t_dwell     Total number of seconds for the dwell phase(s)
  t_total     Total number of seconds to run for

optional arguments:
  -h, --help  show this help message and exit
```

## Requirements
* pyvisa (tested with version 1.8)
* pyvisa-py (tested with version 0.2, this is optional depending on how your sourcemeter is attached)

## Examples
```bash
python mppTracker.py GPIB0::24::INSTR 10 120 # GPIB attached sourcemeter
python mppTracker.py TCPIP::192.168.1.54::INSTR 10 120 # ethernet attached sourcemeter
python mppTracker.py USB0::0x1AB1::0x0588::DS1K00005888::INSTR 10 120 # USB attached sourcemeter
python mppTracker.py ASRL::COM3::INSTR 10 120 # rs232 attached 10 sourcemeter
python mppTracker.py ASRL::/dev/ttyUSB0::INSTR 10 120 # rs232 attached sourcemeter
```
