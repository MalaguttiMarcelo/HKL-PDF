from __future__ import annotations

from typing import Optional, Dict, Any

import os
os.environ.setdefault("QT_API", "pyside6")

import numpy as np
from PySide6 import QtWidgets

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except Exception:
    pv = None
    QtInteractor = None
    PYVISTA_AVAILABLE = False


class FastShapeViewer(QtWidgets.QWidget):
    """
    Fast VTK/OpenGL viewer for crystallite-shape preview.

    This replaces Matplotlib 3D for the shape dialog.
    Matplotlib 3D is CPU-rendered and slow for rotation.
    PyVista/VTK keeps geometry in GPU/VTK buffers, so rotation is much faster.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.plotter = None
        self.actors: Dict[str, Any] = {}
        self._camera_initialized = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not PYVISTA_AVAILABLE:
            label = QtWidgets.QLabel(
                "PyVistaQt is not installed.\n\n"
                "Install with:\n"
                "pip install pyvista pyvistaqt vtk qtpy"
            )
            label.setWordWrap(True)
            layout.addWidget(label)
            return

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        try:
            self.plotter.set_background("#F7F7F7")
            self.plotter.enable_depth_peeling()

            try:
                self.plotter.enable_anti_aliasing("ssaa")
            except Exception:
                try:
                    self.plotter.enable_anti_aliasing("fxaa")
                except Exception:
                    pass
                
            try:
                self.plotter.render_window.SetMultiSamples(8)
            except Exception:
                pass
            
            # Remove the flat default lighting and create a VESTA-like
            # camera-following light setup.
            try:
                self.plotter.remove_all_lights()
            except Exception:
                pass
            
            try:
                key_light = pv.Light(
                    light_type="headlight",
                    color="white",
                    intensity=0.90,
                )
                self.plotter.add_light(key_light)
            except Exception:
                pass
            
            try:
                fill_light = pv.Light(
                    light_type="camera light",
                    color="#FFF1E6",
                    intensity=0.30,
                )
                fill_light.set_direction_angle(
                    35.0,
                    -45.0,
                )
                self.plotter.add_light(fill_light)
            except Exception:
                pass
            
            try:
                rim_light = pv.Light(
                    light_type="camera light",
                    color="#DDE8FF",
                    intensity=0.20,
                )
                rim_light.set_direction_angle(
                    -35.0,
                    135.0,
                )
                self.plotter.add_light(rim_light)
            except Exception:
                pass
            
            # Screen-space ambient occlusion helps separate atoms that
            # are close to one another.
            try:
                self.plotter.enable_ssao(
                    radius=1.5,
                    bias=0.01,
                    kernel_size=128,
                    blur=True,
                )
            except Exception:
                pass
            
        except Exception:
            pass

    def clear_scene(self) -> None:
        if self.plotter is None:
            return

        try:
            self.plotter.clear()
            self.actors.clear()
            self._camera_initialized = False
        except Exception:
            pass

    def _remove_actor(self, key: str) -> None:
        if self.plotter is None:
            return

        actor = self.actors.pop(key, None)

        if actor is None:
            return

        try:
            if isinstance(actor, (list, tuple)):
                for a in actor:
                    try:
                        self.plotter.remove_actor(a)
                    except Exception:
                        pass
            else:
                self.plotter.remove_actor(actor)
        except Exception:
            pass

    def _maybe_reset_camera(self, force: bool = False) -> None:
        if self.plotter is None:
            return

        if force or not self._camera_initialized:
            try:
                self.plotter.reset_camera()
                self._camera_initialized = True
            except Exception:
                pass

    def render(self) -> None:
        if self.plotter is None:
            return

        try:
            self.plotter.render()
        except Exception:
            pass

    def set_atoms(
        self,
        points: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        *,
        visible: bool = True,
        point_size: float = 8.0,
        render_as_spheres: bool = True,
        reset_camera: bool = False,
        use_sphere_glyphs: bool = True,
        sphere_radius: float = 0.42,
        sphere_theta_resolution: int = 24,
        sphere_phi_resolution: int = 18,
        show_silhouette: bool = True,
    ) -> None:
        """
        Add or update an atom cloud.

        For ordinary structures, real sphere glyphs are used. They have smooth
        surface normals, Phong lighting, a bright front/center, darker borders,
        and an optional thin black silhouette.

        Very large structures can use the point-sphere fallback by setting
        use_sphere_glyphs=False.
        """
        if self.plotter is None:
            return

        self._remove_actor("atoms")

        if not visible:
            self.render()
            return

        points = np.asarray(
            points,
            dtype=np.float32,
        )

        if (
            points.ndim != 2
            or points.shape[1] != 3
            or points.shape[0] == 0
        ):
            self.render()
            return

        cloud = pv.PolyData(points)

        rgb_valid = False

        if rgb is not None:
            rgb = np.asarray(rgb)

            if (
                rgb.ndim == 2
                and rgb.shape[1] == 3
                and rgb.shape[0] == points.shape[0]
            ):
                if rgb.dtype != np.uint8:
                    rgb = np.clip(
                        rgb,
                        0,
                        255,
                    ).astype(np.uint8)

                cloud["rgb"] = rgb
                rgb_valid = True

        actor = None

        # ---------------------------------------------------------
        # Preferred rendering: actual sphere geometry.
        #
        # This gives proper smooth shading and a much more VESTA-like
        # three-dimensional appearance.
        # ---------------------------------------------------------
        if use_sphere_glyphs:
            try:
                sphere = pv.Sphere(
                    radius=max(float(sphere_radius), 1.0e-6),
                    theta_resolution=max(
                        int(sphere_theta_resolution),
                        12,
                    ),
                    phi_resolution=max(
                        int(sphere_phi_resolution),
                        8,
                    ),
                )

                try:
                    sphere = sphere.compute_normals(
                        point_normals=True,
                        cell_normals=False,
                        split_vertices=False,
                        auto_orient_normals=True,
                        consistent_normals=True,
                        inplace=False,
                    )
                except Exception:
                    pass

                glyphs = cloud.glyph(
                    geom=sphere,
                    scale=False,
                    orient=False,
                )

                mesh_kwargs = {
                    "smooth_shading": True,
                    "ambient": 0.12,
                    "diffuse": 0.82,
                    "specular": 0.70,
                    "specular_power": 45.0,
                    "show_edges": False,
                }

                if show_silhouette:
                    mesh_kwargs["silhouette"] = {
                        "color": "#202020",
                        "line_width": 1.0,
                        "opacity": 0.55,
                    }

                if rgb_valid and "rgb" in glyphs.array_names:
                    actor = self.plotter.add_mesh(
                        glyphs,
                        scalars="rgb",
                        rgb=True,
                        preference="point",
                        **mesh_kwargs,
                    )
                else:
                    actor = self.plotter.add_mesh(
                        glyphs,
                        color="royalblue",
                        **mesh_kwargs,
                    )

            except TypeError:
                # Some older PyVista versions do not accept the silhouette
                # keyword. Retry without it.
                try:
                    if rgb_valid and "rgb" in glyphs.array_names:
                        actor = self.plotter.add_mesh(
                            glyphs,
                            scalars="rgb",
                            rgb=True,
                            preference="point",
                            smooth_shading=True,
                            ambient=0.12,
                            diffuse=0.82,
                            specular=0.70,
                            specular_power=45.0,
                            show_edges=False,
                        )
                    else:
                        actor = self.plotter.add_mesh(
                            glyphs,
                            color="royalblue",
                            smooth_shading=True,
                            ambient=0.12,
                            diffuse=0.82,
                            specular=0.70,
                            specular_power=45.0,
                            show_edges=False,
                        )

                except Exception:
                    actor = None

            except Exception:
                actor = None

        # ---------------------------------------------------------
        # Fast fallback for very large atom clouds.
        # ---------------------------------------------------------
        if actor is None:
            try:
                if rgb_valid:
                    actor = self.plotter.add_points(
                        cloud,
                        scalars="rgb",
                        rgb=True,
                        point_size=float(point_size),
                        render_points_as_spheres=bool(render_as_spheres),
                        ambient=0.12,
                        diffuse=0.82,
                        specular=0.55,
                        specular_power=35.0,
                    )
                else:
                    actor = self.plotter.add_points(
                        cloud,
                        color="royalblue",
                        point_size=float(point_size),
                        render_points_as_spheres=bool(render_as_spheres),
                        ambient=0.12,
                        diffuse=0.82,
                        specular=0.55,
                        specular_power=35.0,
                    )

            except TypeError:
                try:
                    if rgb_valid:
                        actor = self.plotter.add_points(
                            cloud,
                            scalars="rgb",
                            rgb=True,
                            point_size=float(point_size),
                            render_points_as_spheres=True,
                        )
                    else:
                        actor = self.plotter.add_points(
                            cloud,
                            color="royalblue",
                            point_size=float(point_size),
                            render_points_as_spheres=True,
                        )

                except Exception:
                    actor = None

        if actor is not None:
            try:
                prop = actor.GetProperty()

                prop.LightingOn()
                prop.SetInterpolationToPhong()
                prop.SetAmbient(0.12)
                prop.SetDiffuse(0.82)
                prop.SetSpecular(0.70)
                prop.SetSpecularPower(45.0)

            except Exception:
                pass

            self.actors["atoms"] = actor

        self._maybe_reset_camera(
            force=reset_camera,
        )
        self.render()

    def set_unit_cell(
        self,
        corners: np.ndarray,
        *,
        visible: bool = True,
        color: str = "black",
        line_width: float = 2.0,
    ) -> None:
        """
        Draw unit cell from its 8 corner coordinates.

        corners order:
            0 O
            1 a
            2 b
            3 c
            4 a+b
            5 a+c
            6 b+c
            7 a+b+c
        """
        if self.plotter is None:
            return

        self._remove_actor("unit_cell")

        if not visible:
            self.render()
            return

        corners = np.asarray(corners, dtype=np.float32)

        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5),
            (2, 4), (2, 6),
            (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 7),
        ]

        # Build one PolyData line mesh instead of 12 separate actors.
        lines = []
        for i, j in edges:
            lines.extend([2, i, j])

        mesh = pv.PolyData()
        mesh.points = corners
        mesh.lines = np.asarray(lines, dtype=np.int64)

        actor = self.plotter.add_mesh(
            mesh,
            color=color,
            line_width=float(line_width),
            render_lines_as_tubes=True,
            lighting=True,
        )

        self.actors["unit_cell"] = actor
        self.render()

    def set_axis(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        *,
        visible: bool = True,
        color: str = "red",
        line_width: float = 5.0,
    ) -> None:
        if self.plotter is None:
            return

        self._remove_actor("axis")

        if not visible:
            self.render()
            return

        line = pv.Line(np.asarray(p0, dtype=float), np.asarray(p1, dtype=float))

        actor = self.plotter.add_mesh(
            line,
            color=color,
            line_width=float(line_width),
        )

        self.actors["axis"] = actor
        self.render()


    def set_vector(
        self,
        key: str,
        p0: np.ndarray,
        p1: np.ndarray,
        *,
        visible: bool = True,
        color: str = "red",
        label: str = "",
        line_width: float = 5.0,
        label_size: int = 12,
    ) -> None:
        """
        Draw a simple vector as a line from p0 to p1, with optional label.

        This is intentionally NOT an arrow.
        """
        if self.plotter is None:
            return

        self._remove_actor(key)

        if not visible:
            self.render()
            return

        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)

        if np.linalg.norm(p1 - p0) <= 1e-12:
            self.render()
            return

        actors = []

        try:
            line = pv.Line(p0, p1)

            line_actor = self.plotter.add_mesh(
                line,
                color=color,
                line_width=float(line_width),
            )

            actors.append(line_actor)

            if label:
                label_actor = self.plotter.add_point_labels(
                    np.asarray([p1], dtype=float),
                    [str(label)],
                    font_size=int(label_size),
                    point_size=0,
                    shape_opacity=0.0,
                    text_color=color,
                    always_visible=True,
                )

                actors.append(label_actor)

            self.actors[key] = actors

        except Exception:
            pass

        self.render()


    def set_cylinder_outline(
        self,
        *,
        center: np.ndarray,
        axis: np.ndarray,
        radius: float,
        height: float,
        visible: bool = True,
        color: str = "dodgerblue",
        opacity: float = 0.18,
    ) -> None:
        if self.plotter is None:
            return

        self._remove_actor("shape_outline")

        if not visible:
            self.render()
            return

        center = np.asarray(center, dtype=float)
        axis = np.asarray(axis, dtype=float)

        n = np.linalg.norm(axis)

        if n <= 1e-12:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / n

        radius = float(radius)
        height = float(height)

        actors = []

        # ---------------------------------------------------------
        # Transparent cylinder surface
        # ---------------------------------------------------------
        try:
            cyl = pv.Cylinder(
                center=center,
                direction=axis,
                radius=radius,
                height=height,
                resolution=128,
            )

            surf_actor = self.plotter.add_mesh(
                cyl,
                color=color,
                opacity=float(opacity),
                smooth_shading=True,
                show_edges=False,
            )

            actors.append(surf_actor)

        except Exception:
            pass

        # ---------------------------------------------------------
        # Explicit black outline:
        #   - top circle
        #   - bottom circle
        #   - vertical side generators
        # ---------------------------------------------------------
        try:
            # Build a perpendicular basis to the cylinder axis.
            test = np.array([1.0, 0.0, 0.0], dtype=float)

            if abs(float(np.dot(test, axis))) > 0.90:
                test = np.array([0.0, 1.0, 0.0], dtype=float)

            e1 = test - np.dot(test, axis) * axis
            e1 /= max(np.linalg.norm(e1), 1e-12)

            e2 = np.cross(axis, e1)
            e2 /= max(np.linalg.norm(e2), 1e-12)

            half_h = 0.5 * height

            top_center = center + half_h * axis
            bot_center = center - half_h * axis

            theta = np.linspace(0.0, 2.0 * np.pi, 129)

            top_pts = (
                top_center[None, :]
                + radius * np.cos(theta)[:, None] * e1[None, :]
                + radius * np.sin(theta)[:, None] * e2[None, :]
            )

            bot_pts = (
                bot_center[None, :]
                + radius * np.cos(theta)[:, None] * e1[None, :]
                + radius * np.sin(theta)[:, None] * e2[None, :]
            )

            points = []
            lines = []

            def add_polyline(pts):
                start = len(points)
                points.extend(pts.tolist())
                lines.extend([len(pts)] + list(range(start, start + len(pts))))

            # top and bottom black rings
            add_polyline(top_pts)
            add_polyline(bot_pts)

            # vertical black side lines
            for th in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
                p_top = top_center + radius * np.cos(th) * e1 + radius * np.sin(th) * e2
                p_bot = bot_center + radius * np.cos(th) * e1 + radius * np.sin(th) * e2
                add_polyline(np.asarray([p_bot, p_top], dtype=float))

            line_mesh = pv.PolyData()
            line_mesh.points = np.asarray(points, dtype=float)
            line_mesh.lines = np.asarray(lines, dtype=np.int64)

            edge_actor = self.plotter.add_mesh(
                line_mesh,
                color="black",
                line_width=3.0,
                render_lines_as_tubes=True,
            )

            actors.append(edge_actor)

        except Exception:
            pass

        self.actors["shape_outline"] = actors
        self.render()


    def set_sphere_outline(
        self,
        *,
        center: np.ndarray,
        radius: float,
        visible: bool = True,
        color: str = "dodgerblue",
        opacity: float = 0.16,
    ) -> None:
        if self.plotter is None:
            return

        self._remove_actor("shape_outline")

        if not visible:
            self.render()
            return

        center = np.asarray(center, dtype=float)
        radius = float(radius)

        actors = []

        try:
            sph = pv.Sphere(
                radius=radius,
                center=center,
                theta_resolution=96,
                phi_resolution=48,
            )

            surf_actor = self.plotter.add_mesh(
                sph,
                color=color,
                opacity=float(opacity),
                smooth_shading=True,
                show_edges=False,
            )

            actors.append(surf_actor)

            wire_actor = self.plotter.add_mesh(
                sph,
                style="wireframe",
                color="black",
                line_width=1.5,
                opacity=0.55,
            )

            actors.append(wire_actor)

        except Exception:
            pass

        self.actors["shape_outline"] = actors
        self.render()


    def set_box_outline(
        self,
        bounds: tuple[float, float, float, float, float, float],
        *,
        visible: bool = True,
        color: str = "dodgerblue",
        opacity: float = 0.12,
    ) -> None:
        """
        Axis-aligned box. Useful as a fallback prism preview.
        """
        if self.plotter is None:
            return

        self._remove_actor("shape_outline")

        if not visible:
            self.render()
            return

        actors = []

        try:
            box = pv.Box(bounds=bounds)

            surf_actor = self.plotter.add_mesh(
                box,
                color=color,
                opacity=float(opacity),
                show_edges=False,
            )

            actors.append(surf_actor)

            edge_mesh = box.extract_all_edges()

            edge_actor = self.plotter.add_mesh(
                edge_mesh,
                color="black",
                line_width=3.0,
                render_lines_as_tubes=True,
            )

            actors.append(edge_actor)

        except Exception:
            pass

        self.actors["shape_outline"] = actors
        self.render()



    def set_labels(
        self,
        points: np.ndarray,
        labels: list[str],
        *,
        visible: bool = True,
        font_size: int = 10,
    ) -> None:
        """
        Labels are expensive. Use only for small visible atom counts.
        """
        if self.plotter is None:
            return

        self._remove_actor("labels")

        if not visible:
            self.render()
            return

        points = np.asarray(points, dtype=np.float32)

        if points.size == 0 or not labels:
            self.render()
            return

        try:
            actor = self.plotter.add_point_labels(
                points,
                labels,
                font_size=int(font_size),
                point_size=0,
                shape_opacity=0.0,
                text_color="black",
                always_visible=False,
            )
            self.actors["labels"] = actor
        except Exception:
            pass

        self.render()

    def set_element_legend(
        self,
        element_to_color: Dict[str, str],
        *,
        visible: bool = True,
    ) -> None:
        """
        Display one circular colored legend entry per element type
        in the upper-right corner.
        """
        if self.plotter is None:
            return
    
        self._remove_actor("element_legend")
    
        try:
            self.plotter.remove_legend()
        except Exception:
            pass
        
        if not visible or not element_to_color:
            self.render()
            return
    
        labels = []
    
        for element in sorted(element_to_color.keys()):
            element = str(element).strip()
    
            if not element:
                continue
            
            color = str(
                element_to_color.get(
                    element,
                    "#808080",
                )
            ).strip()
    
            labels.append(
                [
                    element,
                    color,
                ]
            )
    
        if not labels:
            self.render()
            return
    
        number_of_entries = len(labels)
    
        legend_width = 0.17
        legend_height = min(
            max(
                0.055 * number_of_entries + 0.025,
                0.12,
            ),
            0.48,
        )
    
        actor = None
    
        try:
            actor = self.plotter.add_legend(
                labels=labels,
                bcolor="#F0F0F0",
                border=True,
                size=(
                    legend_width,
                    legend_height,
                ),
                loc="upper right",
                face="circle",
                background_opacity=0.94,
            )
    
        except TypeError:
            try:
                actor = self.plotter.add_legend(
                    labels=labels,
                    bcolor="#F0F0F0",
                    border=True,
                    size=(
                        legend_width,
                        legend_height,
                    ),
                    loc="upper right",
                )
    
            except TypeError:
                try:
                    actor = self.plotter.add_legend(
                        labels=labels,
                        bcolor="#F0F0F0",
                        border=True,
                        size=(
                            legend_width,
                            legend_height,
                        ),
                    )
    
                except Exception:
                    actor = None
    
        except Exception:
            actor = None
    
        if actor is not None:
            try:
                text_property = actor.GetEntryTextProperty()
                text_property.SetColor(
                    0.08,
                    0.08,
                    0.08,
                )
                text_property.SetFontSize(14)
                text_property.SetBold(False)
            except Exception:
                pass
            
            try:
                actor.SetPadding(6)
            except Exception:
                pass
            
            self.actors["element_legend"] = actor
    
        self.render()

    def set_arrow(
        self,
        key: str,
        p0: np.ndarray,
        p1: np.ndarray,
        *,
        visible: bool = True,
        color: str = "red",
        label: str = "",
        shaft_radius: float = 0.05,
        tip_radius: float = 0.15,
        tip_length: float = 0.25,
    ) -> None:
        """
        Draw an arrow from p0 to p1 with an optional label.

        key lets us manage axis/base1/base2 separately.
        """
        if self.plotter is None:
            return

        self._remove_actor(key)

        if not visible:
            self.render()
            return

        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)

        direction = p1 - p0
        length = float(np.linalg.norm(direction))

        if length <= 1e-12:
            self.render()
            return

        try:
            arrow = pv.Arrow(
                start=p0,
                direction=direction,
                scale=1.0,
                tip_length=float(tip_length),
                tip_radius=float(tip_radius),
                shaft_radius=float(shaft_radius),
            )

            arrow_actor = self.plotter.add_mesh(
                arrow,
                color=color,
            )

            actors = [arrow_actor]

            if label:
                try:
                    label_actor = self.plotter.add_point_labels(
                        np.asarray([p1], dtype=float),
                        [str(label)],
                        font_size=12,
                        point_size=0,
                        shape_opacity=0.0,
                        text_color=color,
                        always_visible=True,
                    )
                    actors.append(label_actor)
                except Exception:
                    pass

            self.actors[key] = actors

        except Exception:
            # Fallback: line only
            try:
                line = pv.Line(p0, p1)
                line_actor = self.plotter.add_mesh(
                    line,
                    color=color,
                    line_width=5.0,
                )
                self.actors[key] = line_actor
            except Exception:
                pass

        self.render()

    def hide(self, key: str) -> None:
        """
        Remove one actor/group by key.
        """
        self._remove_actor(key)
        self.render()


    def close_viewer(self) -> None:
        """
        Explicitly finalize the PyVista/VTK render window before Qt destroys
        the native window handle.

        This prevents Windows errors like:
            vtkWin32OpenGLRenderWindow: wglMakeCurrent failed
        """
        if self.plotter is None:
            return

        try:
            self.actors.clear()
        except Exception:
            pass

        try:
            self.plotter.clear()
        except Exception:
            pass

        try:
            self.plotter.close()
        except Exception:
            pass

        try:
            interactor = getattr(self.plotter, "interactor", None)
            if interactor is not None:
                try:
                    interactor.Finalize()
                except Exception:
                    pass
                try:
                    interactor.close()
                except Exception:
                    pass
        except Exception:
            pass

        self.plotter = None

    def closeEvent(self, event):
        self.close_viewer()
        super().closeEvent(event)