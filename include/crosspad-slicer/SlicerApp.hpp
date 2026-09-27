#pragma once

#include "lvgl.h"

class App;

/// Builds the app UI as a child of @p parent. Returns the root object.
lv_obj_t* Slicer_create(lv_obj_t* parent, App* app);

/// Tears the app down. Must delete the root with lv_obj_delete_async().
void Slicer_destroy(lv_obj_t* root);
