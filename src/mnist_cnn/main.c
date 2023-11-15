#include "demo_system.h"
#include "mnist_data.h"
#include "timer.h"
#include "gpio.h"
#include "spi.h"
#include <stdio.h>

#include "st7735/lcd_st7735.h"
#include "core/lucida_console_10pt.h"
#include "lcd.h"

// Constants.
enum{
// Pin out mapping.
    LcdCsPin=0,
    LcdRstPin,
    LcdDcPin,
    LcdBlPin,
    SpiSpeedHz = 5 * 100 * 1000,
};

int cnn_Recogn(int, uint32_t *compute_cycles);
static void timer_delay(uint32_t ms);
static void do_experiment(St7735Context *lcd);
static uint32_t spi_write(void *handle, uint8_t *data, size_t len);
static uint32_t gpio_write(void *handle, bool cs, bool dc);
Result lcd_st7735_draw_box(St7735Context *ctx);
Result lcd_st7735_draw_2by2(St7735Context *ctx, LCD_Point centre);


int main(void) {
    timer_init();

    // Set the initial state of the LCD control pins.
    set_output_bit(GPIO_OUT, LcdDcPin, 0x0);
    set_output_bit(GPIO_OUT, LcdBlPin, 0x1);
    set_output_bit(GPIO_OUT, LcdCsPin, 0x0);

    // Init spi driver.
    spi_t spi;
    spi_init(&spi, DEFAULT_SPI, SpiSpeedHz);

    // Reset LCD.
    set_output_bit(GPIO_OUT, LcdRstPin, 0x0);
    timer_delay(150);
    set_output_bit(GPIO_OUT, LcdRstPin, 0x1);

    // Init LCD driver and set the SPI driver.
    St7735Context lcd;
    LCD_Interface interface = {
            .handle = &spi,  // SPI handle.
            .spi_write = spi_write, // SPI write callback.
            .gpio_write = gpio_write, // GPIO write callback.
            .timer_delay = timer_delay, // Timer delay callback.
    };
    lcd_st7735_init(&lcd, &interface);

    // Set the LCD orientation.
    lcd_st7735_set_orientation(&lcd, LCD_Rotate180);

    // Setup text font bitmaps to be used and the colors.
    lcd_st7735_set_font(&lcd, &lucidaConsole_10ptFont);
    lcd_st7735_set_font_colors(&lcd, BGRColorWhite, BGRColorBlack);

    // Clean display with a white rectangle.
    lcd_st7735_clean(&lcd);

    lcd_println(&lcd, "Booting...", alined_center, 7);
    timer_delay(1000);

    do {
        lcd_st7735_clean(&lcd);
        do_experiment(&lcd);
        timer_delay(1000);
    } while(1);
}

static void timer_delay(uint32_t ms){
    // Configure timer to trigger every 1 ms
    timer_enable(50000);
    uint32_t timeout = get_elapsed_time() + ms;
    while(get_elapsed_time() < timeout){ asm volatile ("wfi"); }
    timer_disable();
}

static void do_experiment(St7735Context *lcd)
{
    int retVal = 0;
    char buf[32];
    uint32_t compute_cycles;
    for(int i=0; i<1; i++)
    {
        compute_cycles = 0;
        retVal = cnn_Recogn(i, &compute_cycles);

        snprintf(buf, 32, "Cy: %10ld", compute_cycles);
        lcd_st7735_puts(lcd, (LCD_Point){.x=0, .y=112}, buf);

        snprintf(buf, 32, "Act: %d, Pred: %d", mnist_test_labels[i], retVal);
        lcd_st7735_puts(lcd, (LCD_Point){.x=15, .y=20}, buf);

        lcd_st7735_draw_box(lcd);
        for(int y=0; y<28;y++)
        {
            for(int x=0; x<28; x++)
            {
                if(mnist_test_data[i][y][x] > 0) {
                    lcd_st7735_draw_2by2(lcd, (LCD_Point) {.x=52+(2*x), .y=55+(2*y)});
                }
            }
        }

        timer_delay(1000);
    }
}

static uint32_t spi_write(void *handle, uint8_t *data, size_t len){
    const uint32_t data_sent = len;
    while(len--){
        spi_send_byte_blocking(handle, *data++);
    }
    while((spi_get_status(handle) & spi_status_fifo_empty) != spi_status_fifo_empty);
    return data_sent;
}

static uint32_t gpio_write(void *handle, bool cs, bool dc){
    set_output_bit(GPIO_OUT, LcdDcPin, dc);
    set_output_bit(GPIO_OUT, LcdCsPin, cs);
    return 0;
}

Result lcd_st7735_draw_box(St7735Context *ctx){
    lcd_st7735_fill_rectangle(ctx,
                              (LCD_rectangle){.origin =(LCD_Point){.x = 52, .y = 55} , .width = 56, .height = 56},
                              BGRColorRed);
    return (Result){.code = 0};
}

Result lcd_st7735_draw_2by2(St7735Context *ctx, LCD_Point centre)
{
    lcd_st7735_fill_rectangle(ctx,
                              (LCD_rectangle){.origin =centre , .width = 2, .height = 2},
                              BGRColorBlack);
    return (Result){.code = 0};
}
