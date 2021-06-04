import math

from manim import *


class VectorTrial:
    def __init__(self, tail, head, angle_degrees, percent, arrow_length, vector_stroke_width, arrow_stroke_width):
        self.tail = tail
        self.head = head
        self.angle_degrees = angle_degrees
        self.percent = percent
        self.arrow_length = arrow_length
        self.vector_stroke_width = vector_stroke_width
        self.arrow_stroke_width = arrow_stroke_width

    def myVector(self):
        # head = np.array([2, -1, 0])
        angle = self.angle_degrees * PI / 180
        # arrow_length = 0.3

        # arrow_2 = Arrow(stroke_width=0.5, start=RIGHT, end=LEFT, color=GOLD, tip_shape=ArrowTriangleTip).shift(DOWN)
        # self.play(Create(arrow_2))

        line_tail = np.array(
            [self.tail[0] + self.percent[0] * (self.head[0] - self.tail[0]), self.tail[1] + self.percent[0]
             * (self.head[1] - self.tail[1]), 0])
        line_head = np.array(
            [self.tail[0] + self.percent[1] * (self.head[0] - self.tail[0]), self.tail[1] + self.percent[1]
             * (self.head[1] - self.tail[1]), 0])

        vector_line = Line(line_tail, line_head, stroke_width=self.vector_stroke_width)
        angle_to_axis = angle_of_vector([self.head[0] - self.tail[0], self.head[1] - self.tail[1], self.head[2]
                                         - self.tail[1]])
        first_arrow_angle_to_axis = angle - angle_to_axis
        arrowx = line_head[0] - self.arrow_length * math.cos(first_arrow_angle_to_axis)
        arrowy = line_head[1] + self.arrow_length * math.sin(first_arrow_angle_to_axis)

        arrow_line = Line(line_head, np.array([arrowx, arrowy, 0]), stroke_width=self.arrow_stroke_width)

        second_arrow_angle_to_axis = angle_to_axis + angle
        second_arrowx = line_head[0] - self.arrow_length * math.cos(second_arrow_angle_to_axis)
        second_arrowy = line_head[1] - self.arrow_length * math.sin(second_arrow_angle_to_axis)

        second_arrow_line = Line(line_head, np.array([second_arrowx, second_arrowy, 0]),
                                 stroke_width=self.arrow_stroke_width)

        # self.wait()
        # self.play(Create(vector_line))
        # self.play(Create(arrow_line), Create(second_arrow_line))

        return [vector_line, arrow_line, second_arrow_line]


class PermutePoints:
    def __init__(self, mobjects, what_to_where):
        self.mobjects = mobjects
        self.what_to_where = what_to_where

    def permute(self):
        vectors_fade_in = []
        vectors_fade_out = []
        mobject_moves = []

        for counter in range(len(self.mobjects)):
            point_to_move_to = self.what_to_where[counter].get_center()
            boolean = [point_to_move_to[0] == self.mobjects[counter].get_center()[0],
                       point_to_move_to[1] == self.mobjects[counter].get_center()[1],
                       point_to_move_to[2] == self.mobjects[counter].get_center()[2]]
            if not all(boolean):
                self.mobjects[counter].generate_target()
                self.mobjects[counter].target.move_to(point_to_move_to)
                mobject_moves.append(MoveToTarget(self.mobjects[counter]))
                vector_tail = self.mobjects[counter].get_center()
                vector = VectorTrial(tail=vector_tail, head=point_to_move_to, angle_degrees=30, percent=[0, 0.90],
                                     arrow_length=0.1, vector_stroke_width=2, arrow_stroke_width=0.75).myVector()
                vectors_fade_in.append(FadeIn(*vector))
                vectors_fade_out.append(FadeOut(*vector))
            else:
                list_x_coords = []
                list_y_coords = []
                list_z_coords = []

                for tracker in range(len(self.mobjects)):
                    list_x_coords.append(self.mobjects[tracker].get_center()[0])
                    list_y_coords.append(self.mobjects[tracker].get_center()[1])
                    list_z_coords.append(self.mobjects[tracker].get_center()[2])

                com = [center_of_mass(list_x_coords), center_of_mass(list_y_coords), center_of_mass(list_z_coords)]
                distance = 0.3
                dot = self.mobjects[counter]
                t = 1 + distance / math.sqrt((dot.get_center()[0] - com[0]) * (dot.get_center()[0] - com[0]) +
                                             (dot.get_center()[1] - com[1]) * (dot.get_center()[1] - com[1]))
                circle_center_x = com[0] + t * (dot.get_center()[0] - com[0])
                circle_center_y = com[1] + t * (dot.get_center()[1] - com[1])
                circle_center_z = 0

                circle_center = np.array([circle_center_x, circle_center_y, circle_center_z])

                start_angle = angle_of_vector([dot.get_center()[0] - circle_center_x,
                                               dot.get_center()[1] - circle_center_y,
                                               dot.get_center()[2] - circle_center_z])
                circle = Arc(radius=distance, start_angle=start_angle, angle=2 * PI - PI / 6 + start_angle,
                             arc_center=circle_center)
                mobject_moves.append(Rotating(self.mobjects[counter],
                                              about_point=circle_center, run_time=1))

        return vectors_fade_in, vectors_fade_out, mobject_moves


class CyclePoints(Scene):
    def construct(self):
        points = []
        dots = []

        radius = 2
        number_of_points = 15

        for counter in range(number_of_points):
            p = Circle(radius).point_at_angle(2 * PI * counter / number_of_points)
            points.append(p)
            dot = Dot(points[counter])
            dots.append(dot)

        # self.add(*darray)

        moves = []
        fade = []
        vectors_fade_in = []
        vectors_fade_out = []

        for counter in range(number_of_points):
            pointArray = dots[(counter + 1) % number_of_points].get_center()
            dots[counter].generate_target()
            dots[counter].target.move_to(pointArray)
            moves.append(MoveToTarget(dots[counter]))
            fade.append(FadeIn(dots[counter], run_time=2))
            vector_tail = dots[counter].get_center()
            vector = VectorTrial(tail=vector_tail, head=pointArray, angle_degrees=30, percent=[0, 0.90],
                                 arrow_length=0.1, vector_stroke_width=2, arrow_stroke_width=0.75).myVector()
            vectors_fade_in.append(FadeIn(*vector))
            vectors_fade_out.append(FadeOut(*vector))

        self.wait()
        self.play(*fade)
        self.wait()
        self.play(*vectors_fade_in)
        self.wait()
        # self.play(*vectors_fade_out)
        self.play(*moves, *vectors_fade_out)


class TryPermute(Scene):
    def construct(self):
        points = []
        dots = []
        fade_in_dots = []

        permute = [1, 3, 5, 7, 9, 0, 2, 4, 6, 8, 10, 12, 11, 13, 14]
        what_to_where = []

        radius = 2
        number_of_points = 15

        for counter in range(number_of_points):
            p = Circle(radius).point_at_angle(2 * PI * counter / number_of_points)
            points.append(p)
            dot = Dot(points[counter])
            dots.append(dot)
            fade_in_dots.append(FadeIn(dots[counter], run_time=2))

        for counter in range(number_of_points):
            what_to_where.append(dots[permute[counter]])

        vectors_fade_in, vectors_fade_out, mobject_moves \
            = PermutePoints(mobjects=dots, what_to_where=what_to_where).permute()

        self.wait()
        self.play(*fade_in_dots)
        self.play(*vectors_fade_in)
        self.wait()
        self.play(*vectors_fade_out, *mobject_moves)
