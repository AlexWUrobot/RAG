# FA Report: UART Misclassification Around MPU-6000

## Observation

Several internal debug sessions reported users asking whether MPU-6000 supports UART when they observed serial communication instability in a larger embedded system.

## Internal Finding

MPU-6000 does not support UART as a native device interface. The supported host-side interfaces documented in the datasheet are I2C and SPI.

## Failure Interpretation

If a field log mentions UART together with MPU-6000, treat that as a system-integration mismatch first, not as a supported sensor operating mode.

## Recommended Action

- verify whether the system is actually using I2C or SPI to communicate with the MPU-6000
- check whether UART errors are coming from another subsystem or a bridge MCU
- use the datasheet only to confirm supported interfaces and electrical constraints