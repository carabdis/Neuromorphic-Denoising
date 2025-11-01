#=============================================================================
# (c) 2007-2022  Total Phase, Inc.
#-----------------------------------------------------------------------------
# Project : Windows Driver Installer
# File    : totalphase.ps1
#-----------------------------------------------------------------------------
# Usage is subject to the license terms in LICENSE.txt
#=============================================================================

#=============================================================================
# DEFINITIONS
#=============================================================================
Param([switch]$install, [switch]$uninstall, [switch]$wait)

$version = 'v4.0.0'

if (!$PSScriptRoot) {
    $PSScriptRoot = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
}

$Windows  = $Env:windir
$System32 = 'System32'

if (Test-Path -Path $Windows\Sysnative) {
    $System32 = 'Sysnative'
}

$pnputil = "$Windows\$System32\pnputil.exe"

$win10_1607 = 14393
$win_build  = [int](Get-ItemProperty `
    "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion").CurrentBuild


#=============================================================================
# GENERAL FUNCTIONS
#=============================================================================
function is_admin {
    $user = New-Object Security.Principal.WindowsPrincipal `
        $([Security.Principal.WindowsIdentity]::GetCurrent())

    return $user.IsInRole(
        [Security.Principal.WindowsBuiltinRole]::Administrator
    )
}

function wait_exit {
    if ($wait) {
        $line = Read-Host '::: Press enter to exit ::'
    }

    Exit
}


#=============================================================================
# INSTALL FUNCTIONS
#=============================================================================
function install_inf {
    Param([string]$path)

    $file = Get-Item $PSScriptRoot\$path
    $name = $file.Name

    echo "=== Installing $name"

    if ($win_build -ge $win10_1607) {
        & $pnputil /add-driver $file /install
    }

    else {
        & $pnputil -i -a $file
    }

    if (-not $?) {
        echo ''
        echo 'Install failed.'
        echo ''

        wait_exit
    }

    echo ''
}

function install_item {
    Param([string]$path, [string]$dest)

    $file = Get-Item $PSScriptRoot\$path
    $name = $file.Name

    echo "=== Copying $name to $dest"
    Copy-Item $file $dest
    echo ''
}

function install {
    install_inf winusbtp.inf

    if (Test-Path -Path $Windows\SysWOW64) {
        install_item tpd2xx64.dll $Windows\$System32\tpd2xx.dll
        install_item tpd2xx32.dll $Windows\SysWOW64\tpd2xx.dll
    }

    else {
        install_item tpd2xx32.dll $Windows\$System32\tpd2xx.dll
    }
}


#=============================================================================
# UNINSTALL FUNCTIONS
#=============================================================================
function uninstall_inf {
    Param([string]$path)

    $files = Get-ChildItem $path

    foreach ($file in $files) {
        $match = Get-Content $file |
            Select-String -pattern '%TotalPhase%' -SimpleMatch

        if ($match) {
            echo "=== Uninstalling $file"

            if ($win_build -ge $win10_1607) {
                & $pnputil /delete-driver $file.name /uninstall /force
            }

            else {
                & $pnputil -f -d $file.name
            }

            echo ''
        }
    }
}

function uninstall_item {
    Param([string]$path)

    $items = Get-ChildItem $path

    foreach ($item in $items) {
        $item = $item -replace 'HKEY_LOCAL_MACHINE', 'HKLM:'

        if ($item) {
            echo "=== Removing $item"
            Remove-Item -Path $item -Force
            echo ''
        }
    }
}

function uninstall_dev {
    Param([string]$id)

    if ($win_build -lt $win10_1607) {
        if (Test-Path -Path $Windows\SysWOW64) {
            & "$PSScriptRoot\uninstall64.exe"
        }
        else {
            & "$PSScriptRoot\uninstall32.exe"
        }

        return
    }

    & $pnputil /enum-devices /class USB                |
        Where-Object   { $_ -like '*' + $id }          |
        Foreach-Object { $_ -replace '^.+USB', 'USB' } |
        Foreach-Object {
            $dev = $_
            echo "=== Removing device $dev"
            & $pnputil /remove-device $dev /subtree
        }
}

function uninstall {
    uninstall_inf  $Windows\INF\oem*.inf

    uninstall_item HKLM:\SYSTEM\CurrentControlSet\Control\usbflags\0403E0D0*
    uninstall_item HKLM:\SYSTEM\CurrentControlSet\Control\usbflags\1679*

    uninstall_dev  USB\VID_0403*PID_E0D0\*
    uninstall_dev  USB\VID_1679*PID_????\*

    uninstall_item $Windows\$System32\tpd2xx*.dll
    uninstall_item $Windows\SysWOW64\tpd2xx*.dll

    # Version 2.x
    uninstall_item $Windows\$System32\Drivers\tpdibus*.sys
}


#=============================================================================
# MAIN PROGRAM
#=============================================================================
if (-not (is_admin)) {
    echo 'This script must be run as Administrator'
    wait_exit
}

echo ''
echo "Total Phase USB Driver Installer $version"
echo '(c) 2007-2022 Total Phase, Inc.  All rights reserved.'
echo ''
echo 'Please unplug Total Phase devices before proceeding.'
echo ''

if (!$install -and !$uninstall) {
    echo '1) Install USB drivers and library'
    echo ''
    echo '2) Uninstall USB drivers and library'
    echo ''

    $option = Read-Host 'Option'
    echo ''

    switch ($option) {
        "1" { $install = $true }
        "2" { $uninstall = $true }

        default {
            echo "unknown option: $option"
        }
    }
}

if ($install) {
    echo '==='
    echo '=== Installing USB Driver and Library'
    echo '==='
    echo ''

    uninstall
    install

    echo 'Install complete.'
    echo ''
}

if ($uninstall) {
    echo '==='
    echo '=== Uninstalling Existing Drivers'
    echo '==='
    echo ''

    uninstall

    echo 'Uninstall complete.'
    echo ''
}

wait_exit
