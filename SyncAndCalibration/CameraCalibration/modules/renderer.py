import vtk
import numpy as np
import cv2
from modules.parser import *
class Renderer:
    colors = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Navy', 'Purple', 'Cyan', 'crimson', 'gold', 'maroon', 'mediumpurple', 'pink', 'silver', 'blackboard', 'yellowgreen']
    cams = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

    @classmethod
    def render_camera_scene(cls, work_path, configs):
        window_w = 4500
        window_h = 3000
        window = vtk.vtkRenderWindow()
        vtk_renderer = vtk.vtkRenderer()

        win_size_scaler = 0.25
        window.SetSize(int(win_size_scaler * configs.image_shape[1]), int(win_size_scaler * configs.image_shape[0]))
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)

        # Define viewport ranges
        window.AddRenderer(vtk_renderer)

        """ 
        ------------------------
        draw here
        ------------------------
        """
        for cam_idx in range(configs.num_cams):
            cls.__render_camera(vtk_renderer, work_path, cam_idx, configs)

        # floor
        floor_w = 3500
        floor_h = 2500
        color = 'azure'
        cls.__draw_floor(vtk_renderer, floor_w, floor_h, color)

        # global axis
        axis_length = 1000  # 1m
        cls.__draw_axis(vtk_renderer, np.identity(3), [0, 0, 0], 3, axis_length)
        vtk_renderer.SetBackground(1, 1, 1)
        """ 
        ------------------------
        """
        # set camera positions
        up_vec = np.array([0, 0, 1])
        vtk_renderer.ResetCamera()
        vtk_camera = vtk.vtkCamera()
        vtk_camera.SetPosition(0, window_h, window_w)
        vtk_camera.SetFocalPoint(0, 0, 0)
        vtk_camera.SetViewUp(up_vec[0], up_vec[1], up_vec[2])
        # vtk_camera.OrthogonalizeViewUp()
        vtk_renderer.SetActiveCamera(vtk_camera)

        interactor.Initialize()
        window.Render()
        interactor.Start()


    @classmethod
    def __render_camera(cls, vtk_renderer, work_path, cam_idx, configs):
        color_str = cls.colors[cam_idx]

        """
        # xml
        """
        # xml_path = work_path + '\\SingleCalibrations\cam_param_' + cls.cams[cam_idx] + '.xml'
        # print(xml_path)
        # keys = ['M', 'd', 'rvec_se3', 'tvec_se3']
        # param = Parser.load_xml(xml_path, keys)
        # rvec = param['rvec_se3']
        # t = param['tvec_se3'].reshape(3,)
        # M = param['M']
        # focal_length = (M[0, 0] * configs.cam_sensor_shape[1] / configs.image_shape[1] + M[1, 1] * configs.cam_sensor_shape[0] / configs.image_shape[0]) / 2.
        # R, _ = cv2.Rodrigues(rvec)

        """
        # json
        """
        # json_path = work_path + r'\CameraParameters\cam_params.json'
        json_path = work_path + r'\\Triangulation\input\cam_params.json'
        with open(json_path, 'r') as f:
            j = json.load(f)
            cam_params = j['cam_params']
            param = cam_params[str(cam_idx)]
            rvec = np.array(param['rvec']).reshape((3,))
            t = param['tvec']
            fx = param['fx']
            fy = param['fy']

            focal_length = (fx * configs.cam_sensor_shape[1] / configs.image_shape[1] + fy * configs.cam_sensor_shape[0] / configs.image_shape[0]) / 2.
            f.close()
        R, _ = cv2.Rodrigues(rvec)
        t = -R.T.dot(t)
        R = R.T




        # if cam_idx != 4:
        #     xml_path = r'data\SingleCalibrations\cam_param_' + str(4) + '.xml'
        #     cam_param_4 = Parser.load_xml(xml_path, keys)
        #     R4, _ = cv2.Rodrigues(cam_param_4['rvec'])
        #     t4 = cam_param_4['tvec'].reshape(3,)
        #     t4 = -R4.T.dot(t4)
        #     R4 = R4.T
        #
        #     R = R4.dot(R)
        #     t = R4.dot(t) + t4

        wh = configs.cam_sensor_shape[0]
        hh = configs.cam_sensor_shape[1]

        size_scaler = 15.
        wh *= size_scaler
        hh *= size_scaler
        focal_length *= size_scaler


        p0 = np.array([-wh, hh, focal_length]).reshape(3,)
        p1 = np.array([wh, hh, focal_length]).reshape(3,)
        p2 = np.array([wh, -hh, focal_length]).reshape(3,)
        p3 = np.array([-wh, -hh, focal_length]).reshape(3,)
        p4 = np.array([0, 0, 0]).reshape(3,)
        p0 = R.dot(p0) + t
        p1 = R.dot(p1) + t
        p2 = R.dot(p2) + t
        p3 = R.dot(p3) + t
        p4 = R.dot(p4) + t
        points = vtk.vtkPoints()
        points.InsertNextPoint(p0)
        points.InsertNextPoint(p1)
        points.InsertNextPoint(p2)
        points.InsertNextPoint(p3)
        points.InsertNextPoint(p4)

        # camera face1
        tri1 = vtk.vtkTriangle()
        tri1.GetPointIds().SetId(0, 0)
        tri1.GetPointIds().SetId(1, 1)
        tri1.GetPointIds().SetId(2, 2)

        # camera face2
        tri2 = vtk.vtkTriangle()
        tri2.GetPointIds().SetId(0, 0)
        tri2.GetPointIds().SetId(1, 2)
        tri2.GetPointIds().SetId(2, 3)

        # camera side right
        tri3 = vtk.vtkTriangle()
        tri3.GetPointIds().SetId(0, 0)
        tri3.GetPointIds().SetId(1, 4)
        tri3.GetPointIds().SetId(2, 3)
        # camera side left
        tri4 = vtk.vtkTriangle()
        tri4.GetPointIds().SetId(0, 1)
        tri4.GetPointIds().SetId(1, 4)
        tri4.GetPointIds().SetId(2, 2)
        # camera side top
        tri5 = vtk.vtkTriangle()
        tri5.GetPointIds().SetId(0, 0)
        tri5.GetPointIds().SetId(1, 1)
        tri5.GetPointIds().SetId(2, 4)
        # camera side bottom
        tri6 = vtk.vtkTriangle()
        tri6.GetPointIds().SetId(0, 2)
        tri6.GetPointIds().SetId(1, 3)
        tri6.GetPointIds().SetId(2, 4)

        triangles = vtk.vtkCellArray()
        triangles.InsertNextCell(tri1)
        triangles.InsertNextCell(tri2)
        triangles.InsertNextCell(tri3)
        triangles.InsertNextCell(tri4)
        triangles.InsertNextCell(tri5)
        triangles.InsertNextCell(tri6)

        # setup colors (setting the name to "Colors" is nice but not necessary)
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        namedColors = vtk.vtkNamedColors()
        try:
            colors.InsertNextTupleValue(namedColors.GetColor3ub(color_str))
            colors.InsertNextTupleValue(namedColors.GetColor3ub(color_str))
            colors.InsertNextTupleValue(namedColors.GetColor3ub(color_str))
            colors.InsertNextTupleValue(namedColors.GetColor3ub(color_str))
            colors.InsertNextTupleValue(namedColors.GetColor3ub(color_str))
            colors.InsertNextTupleValue(namedColors.GetColor3ub(color_str))
        except AttributeError:
            # For compatibility with new VTK generic data arrays.
            colors.InsertNextTypedTuple(namedColors.GetColor3ub(color_str))
            colors.InsertNextTypedTuple(namedColors.GetColor3ub(color_str))
            colors.InsertNextTypedTuple(namedColors.GetColor3ub(color_str))
            colors.InsertNextTypedTuple(namedColors.GetColor3ub(color_str))
            colors.InsertNextTypedTuple(namedColors.GetColor3ub(color_str))
            colors.InsertNextTypedTuple(namedColors.GetColor3ub(color_str))

        # polydata object
        trianglePolyData = vtk.vtkPolyData()
        trianglePolyData.SetPoints(points)
        trianglePolyData.SetPolys(triangles)
        trianglePolyData.GetCellData().SetScalars(colors)

        # mapper
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(trianglePolyData)
        else:
            mapper.SetInputData(trianglePolyData)

        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # assign actor to the renderer
        vtk_renderer.AddActor(actor)
        axis_actor = cls.__draw_axis(vtk_renderer, R, t, 2, 500)
        return actor, axis_actor

    @classmethod
    def __draw_axis(cls, vtk_renderer, R, t, width, length):
        x = length*np.array([1, 0, 0])
        y = length*np.array([0, 1, 0])
        z = length*np.array([0, 0, 1])

        x = R.dot(x) + t
        y = R.dot(y) + t
        z = R.dot(z) + t

        # Create the polydata where we will store all the geometric data
        linesPolyData = vtk.vtkPolyData()
        # Create a vtkPoints container and store the points in it
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(t)
        pts.InsertNextPoint(x)
        pts.InsertNextPoint(y)
        pts.InsertNextPoint(z)

        # Add the points to the polydata container
        linesPolyData.SetPoints(pts)

        # Create the first line (between Origin and x)
        line_x = vtk.vtkLine()
        line_x.GetPointIds().SetId(0, 0)  # the second 0 is the index of the Origin in linesPolyData's points
        line_x.GetPointIds().SetId(1, 1)  # the second 1 is the index of x in linesPolyData's points

        # Create the second line (between Origin and y)
        line_y = vtk.vtkLine()
        line_y.GetPointIds().SetId(0, 0)  # the second 0 is the index of the Origin in linesPolyData's points
        line_y.GetPointIds().SetId(1, 2)  # 2 is the index of y in linesPolyData's points

        # Create the second line (between Origin and z)
        line_z = vtk.vtkLine()
        line_z.GetPointIds().SetId(0, 0)  # the second 0 is the index of the Origin in linesPolyData's points
        line_z.GetPointIds().SetId(1, 3)  # 2 is the index of z in linesPolyData's points

        # Create a vtkCellArray container and store the lines in it
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line_x)
        lines.InsertNextCell(line_y)
        lines.InsertNextCell(line_z)

        # Add the lines to the polydata container
        linesPolyData.SetLines(lines)

        namedColors = vtk.vtkNamedColors()

        # Create a vtkUnsignedCharArray container and store the colors in it
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        try:
            colors.InsertNextTupleValue(namedColors.GetColor3ub("Red"))
            colors.InsertNextTupleValue(namedColors.GetColor3ub("Green"))
            colors.InsertNextTupleValue(namedColors.GetColor3ub("Blue"))
        except AttributeError:
            # For compatibility with new VTK generic data arrays.
            colors.InsertNextTypedTuple(namedColors.GetColor3ub("Red"))
            colors.InsertNextTypedTuple(namedColors.GetColor3ub("Green"))
            colors.InsertNextTypedTuple(namedColors.GetColor3ub("Blue"))

        # Color the lines.
        # SetScalars() automatically associates the values in the data array passed as parameter
        # to the elements in the same indices of the cell data array on which it is called.
        # This means the first component (red) of the colors array
        # is matched with the first component of the cell array (line 0)
        # and the second component (green) of the colors array
        # is matched with the second component of the cell array (line 1)
        linesPolyData.GetCellData().SetScalars(colors)

        # Setup the visualization pipeline
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(linesPolyData)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(width)

        vtk_renderer.AddActor(actor)
        return actor


    @classmethod
    def __draw_floor(cls, vtk_renderer, w, h, color):
        p0 = np.array([-w, h, 0])
        p1 = np.array([w, h, 0])
        p2 = np.array([w, -h, 0])
        p3 = np.array([-w, -h, 0])

        points = vtk.vtkPoints()
        points.InsertNextPoint(p0)
        points.InsertNextPoint(p1)
        points.InsertNextPoint(p2)
        points.InsertNextPoint(p3)

        # checkboard triangle 1
        tri1 = vtk.vtkTriangle()
        tri1.GetPointIds().SetId(0, 0)
        tri1.GetPointIds().SetId(1, 1)
        tri1.GetPointIds().SetId(2, 2)
        tri2 = vtk.vtkTriangle()
        tri2.GetPointIds().SetId(0, 0)
        tri2.GetPointIds().SetId(1, 2)
        tri2.GetPointIds().SetId(2, 3)

        triangles = vtk.vtkCellArray()
        triangles.InsertNextCell(tri1)
        triangles.InsertNextCell(tri2)

        # polydata object
        trianglePolyData = vtk.vtkPolyData()
        trianglePolyData.SetPoints(points)
        trianglePolyData.SetPolys(triangles)

        # Create a vtkUnsignedCharArray container and store the colors in it
        namedColors = vtk.vtkNamedColors()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        try:
            colors.InsertNextTupleValue(namedColors.GetColor3ub(color))
            colors.InsertNextTupleValue(namedColors.GetColor3ub(color))
        except AttributeError:
            # For compatibility with new VTK generic data arrays.
            colors.InsertNextTypedTuple(namedColors.GetColor3ub(color))
            colors.InsertNextTypedTuple(namedColors.GetColor3ub(color))
        trianglePolyData.GetCellData().SetScalars(colors)

        # mapper
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(trianglePolyData)
        else:
            mapper.SetInputData(trianglePolyData)

        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetPosition(0, 0, 0)
        vtk_renderer.AddActor(actor)
        return actor