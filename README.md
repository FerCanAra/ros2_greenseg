# greenseg (ROS 2, ament_python)

Implementación de GreenSeg según el artículo (arXiv 2605.25279), ecs. 1–20 y Tabla 2.

## Instalación

```bash
cd ~/ros2_ws/src && cp -r /ruta/greenseg .
sudo apt install python3-scipy ros-$ROS_DISTRO-cv-bridge
cd ~/ros2_ws && colcon build --packages-select greenseg && source install/setup.bash
```

## Uso

```bash
# Robot real (RealSense, 16UC1 en mm)
ros2 launch greenseg greenseg.launch.py

# Simulación (32FC1 en metros también soportado)
ros2 launch greenseg greenseg.launch.py use_sim_time:=true \
    depth_topic:=/camera/depth/image_raw camera_info_topic:=/camera/depth/camera_info
```

No hace falta `depth_filter_node`: el filtro de rango y el `[::2, ::2]` (`pixel_stride: 2`)
están integrados. Si se sigue usando, poner `pixel_stride: 1`.

## Topics

| Topic | Tipo | Contenido |
|---|---|---|
| `/greenseg/obstacles` | PointCloud2 | P_obs^RG (ec. 20) → costmap local |
| `/greenseg/ground` | PointCloud2 | P_ground^RG (ec. 19) |
| `/greenseg/labeled` | PointCloud2 | x y z `label` `rho` `kappa` (0 suelo, 1 obstáculo, 2 above, 3 ruido) |

En RViz, `/greenseg/labeled` con *Color Transformer = Intensity* y canal `label`.

## Pipeline (`greenseg/core.py`)

1. Retroproyección de la profundidad y TF a `base_link` (ec. 1); filtro radial xy (ec. 2).
2. GPF con prior horizontal (ecs. 3–5); clasificación en 4 clases (ec. 6); refinado del plano (ec. 7).
3. Normales PCA en r = 5 cm (ecs. 10–12); |N| < 30 → ruido; ρ ≥ 0,90 (ecs. 13–14); κ ≤ 0,05 (ecs. 15–16).
4. Region Growing desde la semilla más cercana al robot, r_g = 5 cm (ecs. 17–19).
5. Suelo rechazado → obstáculo (ec. 20).

## Decisiones de implementación (no especificadas en el artículo)

- **Semillas del GPF:** puntos más bajos (LPR, 5 %) + 10 cm; 3 iteraciones. Si el plano supera 30°, no hay suelo.
- **Vóxel de 1 cm** y PCA con los ≤ 64 vecinos más cercanos dentro de r: acota el coste en el campo cercano sin cambiar la escala de 5 cm de las normales.
- **Ec. 20:** todo suelo rechazado pasa a obstáculo, salvo los puntos con |N| < 30, que pasan a ruido.
- **Ec. 6:** usa z_i en `base_link` como el artículo. `height_reference: plane` usa la altura sobre el plano ajustado (mejor en pendientes).

## Test offline

```bash
python3 test/test_core.py
```
