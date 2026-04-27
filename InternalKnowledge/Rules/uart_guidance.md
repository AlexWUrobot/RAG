# Engineering Rule: UART Guidance for MPU-6000 Questions

MPU-6000 does not support UART.

When users ask about UART in relation to MPU-6000 or MPU-6050, the system should explicitly say that UART is not a documented interface for these devices and should redirect the answer toward the supported interfaces in the datasheet.

UART is an asynchronous communication method, which is more prone to bit errors or cumulative drift under high-speed transmission than tightly clocked synchronous links.

If a failure-analysis question mentions UART, treat that as an integration hypothesis or architecture mismatch unless an internal FA report states otherwise.