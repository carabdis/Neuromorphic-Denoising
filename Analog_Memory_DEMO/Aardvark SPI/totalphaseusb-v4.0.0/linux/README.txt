                       Total Phase Linux Driver
                       ------------------------

For all products, the Linux driver is based on libusb and there is
no need to install any other drivers on the operating system.

To provide access by normal users to Total Phase USB products, copy
the 99-totalphase.rules file into /etc/udev/rules.d.

By default all Total Phase USB devices are writable by all users.
OWNER, GROUP, MODE may be modified in the file for more selective
permissions.
