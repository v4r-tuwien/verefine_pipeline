# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna
#
# Adapted from...
# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import OpenGL
OpenGL.ERROR_CHECKING = True

# import os
# os.environ['PYOPENGL_PLATFORM'] = "egl"
# import OpenGL.EGL as egl

import ctypes
from ctypes import pointer

import OpenGL.GL as gl


import numpy as np
from glumpy import gloo, app
app.use('glfw')
import time

# Set logging level
from glumpy.log import log
import logging
log.setLevel(logging.WARNING)

import tinyobjloader
from PIL import Image


_vertex_code = """
uniform mat4 u_mv;
uniform mat4 u_mvp;

attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec3 a_color;
attribute vec2 a_texcoord;

varying vec3 v_color;
varying vec2 v_texcoord;
varying float v_eye_depth;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);

    v_color = a_color;
    v_texcoord = a_texcoord;

    // OpenGL Z axis goes out of the screen, so depths are negative
    vec3 v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // Vertex position in eye coords.
    v_eye_depth = -v_eye_pos.z;
}
"""

_fragment_code = """
uniform sampler2D u_texture;
uniform sampler2D u_observation;
uniform int u_use_texture;
uniform int u_mode;
uniform int u_obj_id;

varying vec3 v_color;
varying vec2 v_texcoord;
varying float v_eye_depth;

void main() {
    if(u_mode == 0) {
        if(bool(u_use_texture)) {
            gl_FragColor = vec4(texture2D(u_texture, v_texcoord).xyz, 1.0);
        }
        else
        {
            gl_FragColor = vec4(v_color, 1.0);
        }
    }
    else if(u_mode == 1) {
        gl_FragColor = vec4(v_eye_depth, 0.0, 0.0, 1.0);
    }
    else if(u_mode == 2) {
        gl_FragColor = vec4(v_eye_depth, u_obj_id, 0.0, 1.0);
    }
    else {  // cost in shader
        float ren_depth = float(texture2D(u_texture, v_texcoord).x) * 1000.0;//v_eye_depth*1000.0;
        float obs_depth = float(texture2D(u_observation, v_texcoord).x);
        
        float rendered = 1.0;
        if (ren_depth < 1.0) {
            rendered = 0.0;
        }
        
        float valid = 1.0;
        if (ren_depth < 1.0 || obs_depth < 1.0) {
            valid = 0.0;
        }
        
        float visible = 0.0;
        float delta = 0.0;
        if (valid == 1.0) {
            float difference = ren_depth - obs_depth;
            
            if (difference < 10.0) {
                visible = 1.0;  // only visible -- ren at most [TAU_VIS] behind obs
                
                float distance = abs(difference);
                delta = min(distance / 20.0, 1.0); // difference is always positive
            } 
        }
        
        gl_FragColor = vec4(rendered, visible, delta, 1.0);
    }
}
"""
# depth_obs = observation / 10
# depth_ren = rendered * 1000  # in mm TODO or do this in renderer?
#
# mask = np.logical_and(depth_ren > 0, depth_obs > 0)
# if np.count_nonzero(mask) == 0:  # no valid depth values
#     return 0
#
# mask = np.logical_and(mask,
#                       depth_ren - depth_obs < 10)  # only visible -- ren at most [TAU_VIS] behind obs

class Model:

    _model_id = 0
    _idx_offset = 0

    def __init__(self):
        self.idx = Model._model_id
        Model._model_id += 1
        self.obj_id = -1
        self.vertex_buffer, self.index_buffer, self.texture = None, None, None

    def from_arrays(self, obj_id, pts, normals, colors, faces, texture=None, uvs=None):
        self.obj_id = obj_id

        colors = np.float32(colors[:, :3])
        faces = faces.reshape(faces.shape[0] * faces.shape[1])

        self.texture = texture
        texture_uv = np.zeros((pts.shape[0], 2), np.float32) if uvs is None else uvs

        if colors.max() > 1.0:
            colors /= 255.0  # Color values are expected in range [0, 1]

        # Set the vertex data
        vertices_type = [('a_position', np.float32, 3),
                         ('a_normal', np.float32, 3),
                         ('a_color', np.float32, colors.shape[1]),
                         ('a_texcoord', np.float32, 2)]
        vertices = np.array(list(zip(pts, normals, colors, texture_uv)), vertices_type)

        # prepare buffers
        self.vertex_buffer = vertices  # .view(gloo.VertexBuffer)
        self.index_buffer = (faces.flatten().astype(np.uint32) + Model._idx_offset).view(gloo.IndexBuffer)

        Model._idx_offset += vertices.shape[0]

    def from_obj(self, obj_id, path):
        self.obj_id = obj_id
        print("reading %s" % path)

        reader = tinyobjloader.ObjReader()
        success = reader.ParseFromFile(path)
        if not success:
            print("error loading %s: WARNING=%s, ERROR=%s" % (path, reader.Warning(), reader.Error()))
            return None, None, None

        attributes = reader.GetAttrib()
        pts = np.array(attributes.vertices)
        normals = np.array(attributes.normals)
        colors = np.array(attributes.colors)
        texture_uv = np.array(attributes.texcoords)

        shapes = reader.GetShapes()
        indices = shapes[0].mesh.numpy_indices()  # index v, vn, vt
        index_count = int(indices.shape[0]/3)

        # reshape to vertex_count
        vertex_count = int(pts.shape[0]/3)
        pts = pts.reshape(vertex_count, 3)
        normals = normals.reshape(vertex_count, 3)
        colors = colors.reshape(vertex_count, 3)
        uv_count = int(texture_uv.shape[0] / 2)
        texture_uv = texture_uv.reshape(uv_count, 2)

        # blow up pts, colors and normals to indices size # TODO ideally we just want uv_count vertices; adapt faces accordingly
        pts = pts[indices.reshape(index_count, 3)[:, 0]]
        colors = colors[indices.reshape(index_count, 3)[:, 0]]
        normals = normals[indices.reshape(index_count, 3)[:, 1]]
        texture_uv = texture_uv[indices.reshape(index_count, 3)[:, 2]]

        faces = np.array(list(range(index_count)), dtype=np.uint32).reshape(index_count)

        # Set texture / color of vertices
        materials = reader.GetMaterials()
        base_path = path[:path.rindex("/") + 1]
        texture_path = base_path + materials[0].diffuse_texname
        if texture_path != "":  # TODO or None?
            texture = np.array(Image.open(texture_path.replace("texture_map", "texture_simple")))

            texture = texture.astype(np.uint8)
            texture = np.flipud(texture) # TODO self.texture = np.flipud(texture)
            texture_uv = texture_uv
            colors = np.zeros((pts.shape[0], 3), np.float32)

            self.texture = texture.view(gloo.Texture2D)

        else:
            self.texture = None
            texture_uv = np.zeros((pts.shape[0], 2), np.float32)

            assert (pts.shape[0] == colors.shape[0])
            colors = colors
            if colors.max() > 1.0:
                colors /= 255.0  # Color values are expected in range [0, 1]

        # Set the vertex data
        vertices_type = [('a_position', np.float32, 3),
                         ('a_normal', np.float32, 3),
                         ('a_color', np.float32, colors.shape[1]),
                         ('a_texcoord', np.float32, 2)]
        vertices = np.array(list(zip(pts, normals, colors, texture_uv)), vertices_type)

        # prepare buffers
        self.vertex_buffer = vertices#.view(gloo.VertexBuffer)
        self.index_buffer = (faces.flatten().astype(np.uint32) + Model._idx_offset).view(gloo.IndexBuffer)

        Model._idx_offset += vertices.shape[0]

    def draw(self, program, use_texture=True):
        use_texture = use_texture or self.texture is None
        program['u_use_texture'] = int(use_texture)

        if use_texture:
            program['u_texture'] = self.texture

        program.draw(gl.GL_TRIANGLES, self.index_buffer)


class Plane(Model):

    def __init__(self):
        Model.__init__(self)
        pts = np.array([
            [-2, -2, 0],
            [ 2, -2, 0],
            [ 2,  2, 0],
            [-2,  2, 0]
        ])
        normals = np.array([[0, 0, 1]]*4)
        colors = np.array([[128, 128, 128]]*4)
        faces = np.array([[0, 3, 2], [0, 2, 1]])
        self.from_arrays(0, pts, normals, colors, faces)
        # self.mat_model = np.eye(4, dtype=np.float32)  # From model space to world space


class ScreenQuad(Model):

    def __init__(self, tex):
        Model.__init__(self)
        pts = np.array([
            [-1, -1, 0],
            [ 1, -1, 0],
            [ 1,  1, 0],
            [-1,  1, 0]
        ])
        normals = np.array([[0, 0, 1]]*4)
        colors = np.array([[128, 128, 128]]*4)
        faces = np.array([[0, 3, 2], [0, 2, 1]])
        uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
        self.from_arrays(0, pts, normals, colors, faces, texture=tex, uvs=uvs)


class Renderer:

    _main_egl_ctx = None

    def __init__(self, dataset, width=640, height=480, near=0.01, far=5.0):
        self.width, self.height = width, height
        self.near, self.far = near, far

        # Create window
        # config = app.configuration.Configuration()
        # # Number of samples used around the current pixel for multisample
        # # anti-aliasing (max is 8)
        # # config.samples = 8
        # config.profile = "core"
        # config.major_version = 3
        # config.minor_version = 3
        # self.window = app.Window(640, 480, config=config, visible=False)
        self.window = app.Window(640, 480, visible=False)

        # Frame buffer object
        self.color_depth_buf = np.zeros((480, 640, 4), np.float32).view(gloo.TextureFloat2D)
        # self.stencil_buf = np.zeros((480, 640, 1), np.uint8).view(gloo.Texture2D)
        self.depth_buf = np.zeros((480, 640), np.float32).view(gloo.DepthTexture)
        # self.fbo = gloo.FrameBuffer(color=[self.color_depth_buf, self.stencil_buf], depth=self.depth_buf)
        self.fbo = gloo.FrameBuffer(color=self.color_depth_buf, depth=self.depth_buf)
        self.fbo.activate()

        self.cost_buf = np.zeros((480, 640, 4), np.float32).view(gloo.TextureFloat2D)
        self.fbo2 = gloo.FrameBuffer(color=self.cost_buf)
        # self.fbo2.activate()


# create models
        self.models = [Plane()]
        self.dataset = dataset
        if dataset is not None:
            self.prepare_models(dataset)

        self.screenquad = ScreenQuad(self.color_depth_buf)
        self.models.append(self.screenquad)

        # create single vertex buffer object for all models - bind once, draw according to model indices
        self.vertices = np.array([], dtype=self.models[1].vertex_buffer.dtype)
        for model in self.models:
            self.vertices = np.concatenate((self.vertices, model.vertex_buffer))
        self.vertex_buffer = self.vertices.view(gloo.VertexBuffer)
        for model in self.models:
            model.vertex_buffer = self.vertex_buffer

        # egl related
        # self.eglDpy, self.eglCfg, self.eglSurf, self.eglCtx = None, None, None, None
        # self.setup_egl()
        # self.create_egl_context()

        # Shader
        # self.activate_egl_context()
        self.program = gloo.Program(_vertex_code, _fragment_code)
        # self.deactivate_egl_context()

        self.program['u_texture'] = np.zeros((512, 512, 3), np.float32)
        self.program.bind(self.vertex_buffer)  # is bound once -- saves some time

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glViewport(0, 0, self.width, self.height)
        # Keep the back-face culling disabled because of objects which do not have
        # well-defined surface (e.g. the lamp from the dataset of Hinterstoisser)
        gl.glDisable(gl.GL_CULL_FACE)

        self.runtimes = []

    def setup_egl(self):
        # initialize EGL
        self.eglDpy = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)  # EGLDisplay
        if egl.eglGetError() != int(egl.EGL_SUCCESS):
            print("could not get EGLDisplay")

        major, minor = egl.EGLint(), egl.EGLint()
        egl.eglInitialize(self.eglDpy, pointer(major), pointer(minor))
        if egl.eglGetError() != int(egl.EGL_SUCCESS):
            print("could not initialize EGL")

        # select appropriate configuration
        configAttribs = [
            egl.EGL_SURFACE_TYPE, egl.EGL_PBUFFER_BIT, egl.EGL_BLUE_SIZE, 8,
            egl.EGL_GREEN_SIZE, 8, egl.EGL_RED_SIZE, 8, egl.EGL_DEPTH_SIZE, 24,
            egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_BIT, egl.EGL_NONE
        ]
        configAttribs = (egl.EGLint * len(configAttribs))(*configAttribs)
        self.eglCfg = egl.EGLConfig()
        numConfigs = egl.EGLint()

        egl.eglChooseConfig(self.eglDpy, configAttribs, pointer(self.eglCfg), 1, pointer(numConfigs))
        if egl.eglGetError() != int(egl.EGL_SUCCESS):
            print("could not choose EGLConfig")

        # create surface
        width, height = (self.width, self.height)
        pbufferAttribs = [
            egl.EGL_WIDTH,
            width,
            egl.EGL_HEIGHT,
            height,
            egl.EGL_NONE,
        ]
        pbufferAttribs = (egl.EGLint * len(pbufferAttribs))(*pbufferAttribs)

        self.eglSurf = egl.eglCreatePbufferSurface(self.eglDpy, self.eglCfg, pbufferAttribs)  # EGLSurface
        if egl.eglGetError() != int(egl.EGL_SUCCESS):
            print("could not create EGLSurface")

        # bind API
        egl.eglBindAPI(egl.EGL_OPENGL_API)
        if egl.eglGetError() != int(egl.EGL_SUCCESS):
            print("could not bind OpenGL API")

    def create_egl_context(self):
        if self._main_egl_ctx is None:
            # new context
            self.eglCtx = egl.eglCreateContext(self.eglDpy, self.eglCfg, egl.EGL_NO_CONTEXT, None)
            self._main_egl_ctx = self.eglCtx
        else:
            # setup for context/resource sharing
            # self.eglCtx = egl.eglCreateContext(self.eglDpy, self.eglCfg, self._main_egl_ctx, None)
            self.eglCtx = self._main_egl_ctx
        if egl.eglGetError() != int(egl.EGL_SUCCESS):
            print("could not create EGLContext")

    def activate_egl_context(self):
        egl.eglMakeCurrent(self.eglDpy, self.eglSurf, self.eglSurf, self.eglCtx)
        if egl.eglGetError() != int(egl.EGL_SUCCESS):
            print("could not make context current")

    def deactivate_egl_context(self):
        egl.eglMakeCurrent(self.eglDpy, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, egl.EGL_NO_CONTEXT)
        if egl.eglGetError() != int(egl.EGL_SUCCESS):
            print("could not make context current")

    def destroy_egl_context(self):
        egl.eglDestroyContext(self.eglDpy, self.eglCtx)
        egl.eglTerminate(self.eglDpy)

    def prepare_models(self, dataset):
        if hasattr(dataset, "faces"):
            cld, normals, colors, faces = dataset.cld, dataset.normals, dataset.colors, dataset.faces
            models = list(cld.keys())

            for idx in models:
                model = Model()
                model.from_arrays(idx, cld[idx], normals[idx], colors[idx], faces[idx])
                self.models.append(model)
        else:
            for path in dataset.model_paths:
                model = Model()
                # if hasattr(dataset, "faces"):
                #     model.load_ply(path.replace(".obj", ".ply"))
                # else:
                model.from_obj(1, path)  # TODO get correct obj id
                self.models.append(model)

    # Model-view matrix
    def _compute_model_view(self, model, view):
        return np.dot(model, view)

    # Model-view-projection matrix
    def _compute_model_view_proj(self, model, view, proj):
        return np.dot(np.dot(model, view), proj)

    def set_observation(self, observation):
        self.observed_depth = observation
        self.observation = (observation.astype(np.float32)[::-1, :, :]).view(gloo.TextureFloat2D)  # TODO or do i have the UVs upside down?
        self.program['u_observation'] = self.observation

    def _draw(self, model_ids, model_trafos, mat_view, mat_proj, mode, bbox):

        assert mode in ['color', 'depth', 'depth+seg', 'cost']

        self.fbo.activate()

        # self.activate_egl_context()

        if mode == 'depth' or mode == 'cost':
            dims = 1
            format = gl.GL_RED
            self.program['u_mode'] = 1
        elif mode == 'depth+seg':
            dims = 2
            format = gl.GL_RG
            self.program['u_mode'] = 2
        elif mode == 'color':
            dims = 3
            format = gl.GL_RGB
            self.program['u_mode'] = 0

        # self.program['u_observation'] = self.observation

        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)  # some models (e.g. lamp in Linemod) are one-sided; disable culling to fix render

        # Rendering
        st = time.time()
        for i, (model_id, mat_model) in enumerate(zip(model_ids, model_trafos)):
            model = self.models[model_id]

            self.program['u_mv'] = self._compute_model_view(mat_model, mat_view)
            self.program['u_mvp'] = self._compute_model_view_proj(mat_model, mat_view, mat_proj)
            self.program['u_obj_id'] = model_id

            try:
                model.draw(self.program, use_texture=(mode == 'color'))
                # print("SUCCESS")
            except ValueError as ex:
                print("failed to draw")
                pass
        elapsed_draw = time.time() - st

        # Retrieve the contents of the FBO texture
        st = time.time()
        buffer = np.zeros((self.height, self.width, dims), dtype=np.float32)

        DEBUG_COST = False  # computes using both variants
        if mode != 'cost' or DEBUG_COST:
            gl.glReadPixels(0, 0, self.width, self.height, format, gl.GL_FLOAT, buffer)

            buffer.shape = self.height, self.width, dims
            buffer = buffer[::-1, :]
        elapsed_read = time.time() - st

        st = time.time()
        if mode == 'cost':
            # self.fbo.deactivate()
            self.fbo2.activate()
            self.program['u_texture'] = self.color_depth_buf
            dims = 4
            format = gl.GL_BGRA
            self.program['u_mode'] = 3
            # self.program['u_observation'] = self.observation

            self.program['u_mv'] = np.eye(4)
            self.program['u_mvp'] = np.eye(4)
            self.program['u_obj_id'] = 0
            self.screenquad.draw(self.program, False)

            # st = time.time()
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.cost_buf.handle)
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

            layer = 4
            buf = np.zeros((int(self.height / (2 ** layer)), int(self.width / (2 ** layer)), dims), dtype=np.float32)
            test = gl.glGetTexImage(gl.GL_TEXTURE_2D, layer, format, gl.GL_FLOAT, buf)
            # print(time.time() - st)
            # mipmap_sum = test.sum() * (2 ** (layer * 2))

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

            layer_factor = (2 ** (layer * 2))  # number of pixels that was binned into a single value
            n_visible = test[:, :, 1].sum() * layer_factor
            n_rendered = test[:, :, 2].sum() * layer_factor
            if n_visible == 0 or n_rendered == 0:
                fit = 0  # visibility ratio is 0 or NaN
            else:
                visibility_ratio = n_visible / n_rendered
                delta = (test[:, :, 0].sum() * layer_factor) / n_visible
                fit = visibility_ratio * (1 - delta)

                if DEBUG_COST:
                    print(fit)
        if DEBUG_COST:
            # cost computation (CPU)
            def fit(observation, rendered):
                depth_obs = observation
                depth_ren = rendered * 1000  # in mm

                mask = np.logical_and(depth_ren > 0, depth_obs > 0)
                if np.count_nonzero(mask) == 0:  # no valid depth values
                    return 0

                mask = np.logical_and(mask,
                                      depth_ren - depth_obs < 10)  # only visible -- ren at most [TAU_VIS] behind obs
                dist = np.abs(depth_obs[mask] - depth_ren[mask])
                delta = np.mean(np.min(np.vstack((dist / 20, np.ones(dist.shape[0]))), axis=0))
                visibility_ratio = float(mask.sum()) / float((depth_ren > 0).sum())  # visible / rendered count

                fit = visibility_ratio * (1 - delta)
                if np.isnan(fit):  # invisible object
                    return 0

                return fit

            fitness = fit(self.observed_depth, buffer)
            if DEBUG_COST:
                print(fitness)
        elapsed_cost = time.time() - st

        self.runtimes.append([elapsed_draw, elapsed_read, elapsed_cost])

        # self.deactivate_egl_context()

        if mode == "cost":
            return buffer, fit
        else:
            return buffer

    def compute_view(self, cam, topview=False):
        """
        Adapted from Bullet3 PhysicsClientC_API.cpp
        :param extrinsics:
        :param topview:
        :return:
        """
        R, t = cam[:3, :3], cam[:3, 3]

        # camera coord axes
        if topview:  # look down from 1m above origin towards origin
            # TODO could give a focus point here
            eye = np.matrix([0, 0, 1]).T
            forward = eye - np.matrix([0, 0, 0]).T
            up = R[:, 2]  # TODO could also align this to camera z
        else:
            eye = t
            forward = R * np.matrix([0, 0, 1]).T
            up = R * np.matrix([0, -1, 0]).T

        # normalize
        f = forward / np.linalg.norm(forward)
        u = up / np.linalg.norm(up)
        s = np.cross(f.T, u.T).T  # TODO what is s?
        s /= np.linalg.norm(s)

        # view matrix
        view = np.matrix(np.eye(4))
        view[:3, 0] = s
        view[:3, 1] = u
        view[:3, 2] = -f

        view[3, 0] = -np.dot(s.T, eye)
        view[3, 1] = -np.dot(u.T, eye)
        view[3, 2] = np.dot(f.T, eye)

        # -> back to world
        return view

    def compute_proj(self, intrinsics, perspective=True):
        if perspective:  # perspective
            width, height = self.width, self.height
            near, far = self.near, self.far
            fx = intrinsics[0, 0]  # focal length
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]  # principal point
            cy = intrinsics[1, 2]
            s = intrinsics[0, 1]  # skew
            q = -(far + near) / (far - near)  # near plane
            qn = -2 * (far * near) / (far - near)  # far plane
            x0, y0 = 0, 0  # origin
            mat_proj = np.array([[2 * fx / width, -2 * s / width, (-2 * cx + width + 2 * x0) / width, 0],
                                 [0, 2 * fy / height, (2 * cy - height + 2 * y0) / height, 0],
                                 [0, 0, q, qn],
                                 [0, 0, -1, 0]]).T
        else:  # orthographic
            r = 0.5
            l = -r
            t = r * (self.height / self.width)
            b = -t
            far = 5.0
            near = 0.001
            mat_proj = np.array([[2 / (r - l), 0, 0, -(r + l) / (r - l)],
                                 [0, 2 / (t - b), 0, -(t + b) / (t - b)],
                                 [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                                 [0, 0, 0, 1]]).T
        return mat_proj

    #-------------------------------------------------------------------------------
    def render(self, model_ids, model_trafos, extrinsics, intrinsics, mode='depth+seg',
               perspective=True, top=False, bbox=None):

        t_start = time.time()

        # model trafo given in camera coordinates -> to world, transpose to be row major
        R = extrinsics[:3, :3].T
        t = -R * extrinsics[:3, 3]
        mat_cam = np.matrix(np.eye(4))
        mat_cam[:3, :3], mat_cam[:3, 3] = R, t
        model_trafos = [(mat_cam * mat_model).T for mat_model in model_trafos]

        # gl view matrix from camera matrix
        mat_view = self.compute_view(mat_cam, topview=top)

        # projection matrix
        mat_proj = self.compute_proj(intrinsics, perspective)

        # global rgb, depth, seg
        # rgb, depth, seg = None, None, None

        # @self.window.event
        # def on_draw(dt):
        #     self.window.clear()

            # TODO is it much faster to have a single draw and multiple read-backs for e.g. color+depth+seg?

        rgb, depth, seg = None, None, None
        if mode == 'depth' or mode == 'cost':  # depth-only
            buffer = self._draw(model_ids, model_trafos, mat_view, mat_proj, mode, bbox)
            if mode == 'cost':
                buffer, cost = buffer
            depth = buffer[:, :, 0].astype(np.float32)
        else:
            if 'color' in mode:
                buffer = self._draw(model_ids, model_trafos, mat_view, mat_proj, 'color', bbox)
                rgb = np.round(buffer[:, :, :3] * 255).astype(np.uint8)  # Convert to [0, 255]
            if 'depth+seg' in mode:
                buffer = self._draw(model_ids, model_trafos, mat_view, mat_proj, 'depth+seg', bbox)
                depth = buffer[:, :, 0].astype(np.float32)
                seg = buffer[:, :, 1].astype(np.uint8)

        # app.run(framecount=0)  # The on_draw function is called framecount+1 times

        # self.runtimes.append((time.time() - t_start))
        # print("rendering took %ims" % (self.runtimes[-1]*1000))

        # Set output
        #---------------------------------------------------------------------------
        if mode == 'depth':
            return None, depth, None
        elif mode == 'cost':
            return [None, depth, None], cost
        elif mode == 'color':
            return rgb, None, None
        elif mode == 'color+depth+seg':
            return rgb, depth, seg
        elif mode == 'depth+seg':
            return None, depth, seg

# import trimesh
# import pyrender
# import os
# from bop_toolkit_lib import inout
# import numpy as np
#
#
# class RendererPy:
#
#     def __init__(self, dataset, width, height):
#         self.dataset = dataset
#         self.width = width
#         self.height = height
#
#         cfg_path_detection = "/verefine/3rdparty/Pix2Pose/ros_kinetic/ros_config.json"
#         cfg = inout.load_json(cfg_path_detection)
#
#         self.model_scale = cfg['model_scale']
#         self.obj_models = []
#         for t_id in range(1, 22):
#             ply_fn = os.path.join(cfg['model_dir'], "obj_%0.6d.ply" % t_id)
#             obj_model = trimesh.load_mesh(ply_fn)
#             obj_model.vertices = obj_model.vertices * self.model_scale
#             mesh = pyrender.Mesh.from_trimesh(obj_model)
#             self.obj_models.append(mesh)
#
#     def render(self, model_ids, model_trafos, extrinsics, intrinsics, mode=None):
#         scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(1, 1, 1))
#         camera = pyrender.IntrinsicsCamera(intrinsics[0, 0], intrinsics[1, 1],
#                                            intrinsics[0, 2], intrinsics[1, 2])
#         camera_pose = np.array([[1.0, 0, 0.0, 0],
#                                 [0.0, -1.0, 0.0, 0],
#                                 [0.0, 0.0, -1, 0],
#                                 [0.0, 0.0, 0.0, 1.0]])
#         scene.add(camera, pose=camera_pose)
#
#         for model_id, model_trafo in zip(model_ids, model_trafos):
#             scene.add(self.obj_models[model_id-1], pose=np.array(model_trafo))
#
#         scene.add(pyrender.light.PointLight((0, 1, 0), 1000, range=1000), model_trafo)
#
#         r = pyrender.OffscreenRenderer(self.width, self.height)
#         color, depth = r.render(scene)
#         return color, depth, depth > 0