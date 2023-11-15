add_library(mnist_cnn_lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor/display_drivers/core/lcd_base.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor/display_drivers/core/lucida_console_10pt.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor/display_drivers/core/lucida_console_12pt.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor/display_drivers/st7735/lcd_st7735.c
)

string(APPEND CMAKE_C_FLAGS " -O2")

add_prog(mnist_cnn "main.c;cnn.c;lcd.c")
target_link_libraries(mnist_cnn m)

# pull in core dependencies and additional i2c hardware support
target_link_libraries(mnist_cnn mnist_cnn_lib)
target_include_directories(mnist_cnn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor/display_drivers)
