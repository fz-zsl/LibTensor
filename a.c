#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct PandoraBox {
    int item_count; // Number of items stored in the PandoraBox.
    int *item_size; // Array of item sizes in bytes.
    void *data;     // A type-agnostic pointer to store data.
} Panbox;
void *mycpy(void *dst, const void *src, size_t size) {
    char *tmp_dst = (char*)dst;
    const char *tmp_src = (const char*)src;
    for (size_t i = 0; i < size; i++) tmp_dst[i] = tmp_src[i];
    return dst;
}
void *myset(void *dst, int n, size_t size) {
    char *tmp_dst = (char*)dst; char value = (char)n;
    for (size_t i = 0; i < size; i++) tmp_dst[i] = value;
    return dst;
}
Panbox *create(int item_count, int item_size[]) {
    int num = 0;
    if (item_count < 0) return NULL;  //negative item_count is not supported,
    if (item_count != 0 && item_size == NULL) return NULL; //item_size should not be NULL when item_count is non-zero.
    for (int i = 0; i < item_count; i++) {
        num += item_size[i];
        if (item_size[i] <= 0) return NULL;
    }
    Panbox *box = (Panbox *)malloc(sizeof(Panbox));
    if (box == NULL) return NULL;
    box->item_count = item_count; 
    if (item_count > 0) {
        box->item_size = (int *)calloc(item_count, sizeof(int));
        if (box->item_size == NULL) {
            free(box);
            return NULL;
        }
        mycpy(box->item_size, item_size, sizeof(int) * item_count);
    } else {
        box->item_size = NULL;
        box->data = NULL;
    }
    box->data = calloc(1, num);
    return box;
}
void append(Panbox *panbox, void *value, int width) {
    if (panbox == NULL || value == NULL || width <= 0) return;
    int total_size = 0;
    for (int i = 0; i < panbox->item_count; i++) total_size += panbox->item_size[i];
    int *new_item_size = (int *)realloc(panbox->item_size, sizeof(int) * (panbox->item_count + 1));
    if (new_item_size == NULL) return;
    void *new_data = realloc(panbox->data, total_size + width);
    if (new_data == NULL) {return; }// free(new_item_size);
    panbox->item_size = new_item_size;
    panbox->data = new_data;
    panbox->item_size[panbox->item_count] = width; 
    panbox->item_count++;
    //mycpy(panbox->data + total_size, value, width);
    char *incep = (char *)panbox->data + total_size;
    char *dst = (char *)value;
    for (int i = 0; i < width; i++) incep[i] = dst[i];
}
void write(Panbox *panbox, int item_id, void *value, int width) {
    if (panbox == NULL || item_id < 0) return;
    if (item_id >= panbox->item_count || value == NULL || width <= 0) return;
    if (width > panbox->item_size[item_id]) return;
    int offset = 0;
    for (int i = 0; i < item_id; i++) offset += panbox->item_size[i];
    char *item_start = ((char*)panbox->data) + offset;
    for (int i = 0; i < width; i++) {
        item_start[i] = ((char*)value)[i];
    }
    if (width < panbox->item_size[item_id]) {
        char pad_value = ((char*)value)[width - 1] & 0x80 ? 0xFF : 0x00;
        myset(item_start + width, pad_value, panbox->item_size[item_id] - width);
    }
}
void *read(Panbox *panbox, int item_id) {
    if (panbox == NULL || item_id < 0 || item_id >= panbox->item_count) return NULL;
    int offset = 0;
    for (int i = 0; i < item_id; i++) offset += panbox->item_size[i];
    char *item_start = ((char*)panbox->data) + offset;
    void *copy = malloc(panbox->item_size[item_id]);
    if (copy == NULL) return NULL;
    mycpy(copy, item_start, panbox->item_size[item_id]);
    return copy;
}
void destroy(Panbox *panbox) {
    if (panbox == NULL) return;
    if (panbox->item_size != NULL) free(panbox->item_size);
    if (panbox->data != NULL) free(panbox->data);
    free(panbox);
}
void printc(void *value, int width) {
    if (value == NULL) return;
    if (width <= 0) return;
    int flag = 0;
    for (int i = 0; i < width; i++) {
        unsigned char c = ((unsigned char*)value)[i];
        if (c > 127) {
            printf("- ");
            flag = 1;
        } else if (c > 32 && c != 127) {
            printf("%c ", c);
            flag = 1;
        }
    }
    if (flag) printf("\n");
}
void printx(void *value, int width) {
    if (value == NULL || width <= 0) return;
    unsigned char *p = (unsigned char *)value;
    printf("0x");
    for (int i = width - 1; i >= 0; i--) {
        printf("%02x", p[i]);
    }
    printf("\n");
    return;
}
void hex2byte(void *dst, char *hex) {
    if (dst == NULL || hex == NULL) return; 
    if (hex[0] != '0' || (hex[1] != 'x' && hex[1] != 'X')) return;  
    hex += 2;
    int len = strlen(hex);
    int dst_index = 0;
    if (len % 2 != 0) {
        unsigned char high = 0;
        if (sscanf(hex, "%1hhx", &high) != 1) return;
        ((unsigned char*)dst)[len / 2] = high;
        hex++; len--;
    }
    for (int i = len - 2; i >= 0; i -= 2) {
        unsigned char byte;
        if (sscanf(hex + i, "%2hhx", &byte) != 1) return;
        ((unsigned char*)dst)[dst_index++] = byte;
    }
    return;
}
void show_info(Panbox *p) {
    if (p != NULL) {
        printf("%d\n", p->item_count);
        for (int i = 0; i < p->item_count; i++) {
            printf("%d ", p->item_size[i]);
        }
        printf("\n");
    }
    else {
        printf("NULL\n");
    }
}
Panbox *p = NULL;
int main() {
    freopen("A1.in", "r", stdin);
    freopen("my.out", "w", stdout);
    int T;
    scanf("%d", &T);
    while (T--) {
        char op; 
        scanf(" %c", &op);
        switch (op) {
        case 'C': {
            int item_count;
            scanf("%d", &item_count);
            int *item_size = (int *)malloc(item_count * sizeof(int));
            for (int i = 0; i < item_count; i++) {
                scanf("%d", &item_size[i]);
            }
            p = create(item_count, item_size);
            free(item_size);
            break;
        }
        case 'D': {
            if (p != NULL) {
                for (int i = p->item_count - 1; i >= 0; i--) {
                    void *data = read(p, i);
                    if (data != NULL) {
                        printc(data, p->item_size[i]);
                    }
                    free(data);
                }
            }
            destroy(p);
            break;
        }
        case 'A': {
            int len;
            scanf("%d", &len);
            char *hex_str = (char *)malloc(len + 1);
            scanf("%s", hex_str);
            int num_of_byte = (strlen(hex_str) - 1) / 2;
            void *data = (void *)malloc(num_of_byte);
            hex2byte(data, hex_str);
            append(p, data, num_of_byte);
            free(hex_str);
            free(data);
            break;
        }
        case 'W': {
            int item_id;
            scanf("%d", &item_id);
            int len;
            scanf("%d", &len);
            char *hex_str = (char *)malloc(len + 1);
            scanf("%s", hex_str);
            int num_of_byte = (strlen(hex_str) - 1) / 2;
            void *data = (void *)malloc(num_of_byte);
            hex2byte(data, hex_str);
            write(p, item_id, data, num_of_byte);
            free(hex_str);
            free(data);
            break;
        }
        case 'R': {
            int item_id;
            scanf("%d", &item_id);
            int is_printc;
            scanf("%d", &is_printc);
            void *data = read(p, item_id);
            if (data != NULL) {
                if (is_printc) {
                    printc(data, p->item_size[item_id]);
                }
                else {
                    printx(data, p->item_size[item_id]);
                }
            }
            break;
        }
        case 'Q': {
            show_info(p);
            break;
        }
        default: {
            break;
        }
        }
    }
    return 0;
}