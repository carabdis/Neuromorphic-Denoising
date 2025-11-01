::============================================================================
:: (c) 2007-2022  Total Phase, Inc.
::----------------------------------------------------------------------------
:: Project : Windows Driver Installer
:: File    : SETUP.cmd
::----------------------------------------------------------------------------
:: Usage is subject to the license terms in LICENSE.txt
::============================================================================

@echo off

echo Launching USB driver installation script

timeout /t 5

powershell -Command "&{ Start-Process powershell -Verb RunAs -ArgumentList '-ExecutionPolicy Bypass -File %cd%\totalphase.ps1 %arg% -wait' }"
