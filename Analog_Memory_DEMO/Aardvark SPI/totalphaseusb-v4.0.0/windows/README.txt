                       Windows Driver Installer
                       ------------------------

Introduction
------------
This folder contains Windows drivers for Total Phase products
that require USB drivers.  Either the script installer or the
INF/CAT files may be used to install the drivers.


Automated Installation and Uninstallation
-----------------------------------------
1) Disconnect Total Phase USB devices.

2) Double-click the SETUP.CMD script.

3) Follow the prompts to install or uninstall the drivers.

4) Connect Total Phase USB devices.

This will install the driver INF files for all USB products.

Additionally, the TPD2XX.DLL library will be installed into
the Windows directory to support Aardvark API v5.50 and earlier.


Manual Installation
-------------------
1) Disconnect Total Phase USB devices.

2) Right-click on WINUSBTP.INF and choose Install

3) Connect Total Phase USB devices.

[Optional]
4) Copy TPD2XX32.DLL or TPD2XX64.DLL to the application folder
or the appropriate location in the Windows system folders.

The library must be named TPD2XX.DLL in the destination folder.


Manual Uninstallation
---------------------
Run the following commands in an administrator command prompt.

[Windows 10 1607 and later]
1) Show the installed drivers:
pnputil /enum-drivers

2) Find the Total Phase INF in the list (named like oemNN.inf)

3) Remove the driver:
pnputil /delete-driver oemNN.inf /uninstall /force

4) Show the installed devices:
pnputil /enum-devices /class USB

5) Find the Total Phase instance IDs:
USB\VID_0403&PID_E0D0
USB\VID_1679&PID_NNNN

6) Remove the devices:
pnputil /remove-device INSTANCE /subtree

7) Remove any copies of TPD2XX.DLL from the system.

[Windows 7 and later]
1) Show the installed drivers:
pnputil -e

2) Find the Total Phase INF in the list (named like oemNN.inf)

3) Remove the driver:
pnputil -f -d oemNN.inf

4) Remove the installed devices:
uninstall64  (for 64-bit systems)
uninstall32  (for 32-bit systems)

5) Remove any copies of TPD2XX.DLL from the system.
