
import math
from abc import ABC

import colour
from manim import *
from sympy.ntheory import factorint
import math
import numpy as np
import random


class VectorTrial:
    def __init__(self, tail, head, angle_degrees, percent, arrow_length, vector_stroke_width, arrow_stroke_width,
                 arrow_point_to):
        self.tail = tail
        self.head = head
        self.angle_degrees = angle_degrees
        self.percent = percent
        self.arrow_length = arrow_length
        self.vector_stroke_width = vector_stroke_width
        self.arrow_stroke_width = arrow_stroke_width
        self.arrow_point_to = 0
        self.arrow_point_to = arrow_point_to

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
        angle_to_axis = angle_of_vector([-self.tail[0] + self.arrow_point_to[0], -self.tail[1] + self.arrow_point_to[1],
                                         -self.tail[2] + self.arrow_point_to[2]])
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


class CustomArrowTip(ArrowTip, Triangle, ABC):
    def __init__(self, **kwargs):
        Triangle.__init__(self)
        self.scale(0.05)
        self.set_color(WHITE)
        self.set_fill(color=WHITE, opacity=0)


class SelfArrow:
    def __init__(self, location, center, remaining_angle, arrowtip):
        self.location = location
        self.center = center
        self.remaining_angle = remaining_angle
        self.arrowtip = arrowtip

    def path_to_itself(self):
        radius = math.sqrt((self.location[0] - self.center[0]) * (self.location[0] - self.center[0]) +
                           (self.location[1] - self.center[1]) * (self.location[1] - self.center[1]) +
                           (self.location[2] - self.center[2]) * (self.location[2] - self.center[2]))
        start_angle = angle_of_vector([self.location[0] - self.center[0],
                                       self.location[1] - self.center[1],
                                       self.location[2] - self.center[2]])
        circle = Arc(radius=radius, start_angle=start_angle, angle=2 * PI - self.remaining_angle,
                     arc_center=self.center, stroke_width=2)
        final_angle = start_angle + 2 * PI - self.remaining_angle
        epsilon_angle = 1 * PI / 180
        vector_start_point = np.array([self.center[0] + radius * math.cos(final_angle),
                                       self.center[1] + radius * math.sin(final_angle),
                                       0])
        vector_end_point = np.array([self.center[0] + radius * math.cos(final_angle + epsilon_angle),
                                     self.center[1] + radius * math.sin(final_angle + epsilon_angle),
                                     0])
        arrow = Arrow(
            vector_start_point,
            vector_end_point,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.05,
            tip_shape=CustomArrowTip
        )
        # tip_shape=ArrowTriangleTip #self.arrowtip
        # round(arrow.tip.tip_length, 3)
        return [circle, arrow]


class PermutePoints:
    def __init__(self, mobjects, what_to_where):
        self.mobjects = mobjects
        self.what_to_where = what_to_where

    def permute(self):
        vectors_fade_in = []
        vectors_fade_out = []
        mobject_moves = []
        circ_create = []
        circ_uncreate = []
        tip_create = []
        tip_uncreate = []

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
                vector = Arrow(vector_tail,
                               point_to_move_to,
                               stroke_width=2,
                               max_tip_length_to_length_ratio=0.05,
                               tip_shape=CustomArrowTip
                               )
                vectors_fade_in.append(Create(vector, run_time=1))
                vectors_fade_out.append(FadeOut(vector, run_time=1))
            else:
                list_x_coords = []
                list_y_coords = []
                list_z_coords = []

                for tracker in range(len(self.mobjects)):
                    list_x_coords.append(self.mobjects[tracker].get_center()[0])
                    list_y_coords.append(self.mobjects[tracker].get_center()[1])
                    list_z_coords.append(self.mobjects[tracker].get_center()[2])

                com = [center_of_mass(list_x_coords), center_of_mass(list_y_coords), center_of_mass(list_z_coords)]
                distance = 0.2
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
                rotang = PI / 6
                # circle = Arc(radius=distance, start_angle=start_angle, angle=2 * PI - rotang,
                #             arc_center=circle_center, stroke_width=2)
                mobject_moves.append(Rotating(self.mobjects[counter],
                                              about_point=circle_center, run_time=1))
                arrow_points = np.array(circle_center + [distance * math.cos(-rotang + start_angle),
                                                         distance * math.sin(-rotang + start_angle),
                                                         0])
                # arrow = VectorTrial(tail=arrow_points, head=arrow_points, angle_degrees=60, percent=[0, 0],
                #                    arrow_length=0.1, vector_stroke_width=2, arrow_stroke_width=0.75,
                #                    arrow_point_to=dot.get_center()).myVector()
                selfarrow = SelfArrow(dot.get_center(), circle_center, 45 * PI / 180, CustomArrowTip)
                circle, arrow = selfarrow.path_to_itself()

                circ_time = 0.8
                arrow_time = 1 - circ_time

                circ_create.append(Create(circle, run_time=circ_time))
                circ_uncreate.append(FadeOut(circle, run_time=1))
                tip_create.append(Create(arrow, run_time=arrow_time))
                tip_uncreate.append(FadeOut(arrow, run_time=1))

        return vectors_fade_in, vectors_fade_out, mobject_moves, circ_create, circ_uncreate, tip_create, tip_uncreate


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
                                 arrow_length=0.1, vector_stroke_width=2, arrow_stroke_width=0.75,
                                 arrow_point_to=pointArray).myVector()
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

        vectors_fade_in, vectors_fade_out, mobject_moves, circ_create, circ_uncreate, tip_create, tip_uncreate \
            = PermutePoints(mobjects=dots, what_to_where=what_to_where).permute()

        animationGroup = []

        for i in range(len(circ_create)):
            animationGroup.append(AnimationGroup(circ_create[i], tip_create[i], lag_ratio=1))

        self.wait()
        self.play(*fade_in_dots)
        self.play(*vectors_fade_in, *animationGroup)
        self.wait()
        self.play(*vectors_fade_out, *tip_uncreate, *circ_uncreate, *mobject_moves)
       
 class LCMExplanation(Scene):
    def prime_factorization_in_tex(self, num_to_be_factored):
        tex_string = ''
        is_last_index = False
        for prime in factorint(num_to_be_factored):
            exponent = factorint(num_to_be_factored)[prime]
            if exponent > 1:
                tex_string = tex_string + str(prime) + '^{' + str(factorint(num_to_be_factored)[prime]) + '}\cdot '
            else:
                tex_string = tex_string + str(prime) + '\cdot '

        if tex_string[len(tex_string) - 2: len(tex_string): 1] == 't ':
            tex_string = tex_string[0: len(tex_string) - 6: 1]

        return tex_string

    def moveTo(self, mobject, move_to_location, run_time):
        mobject.generate_target()
        mobject.target.shift(move_to_location)
        return MoveToTarget(mobject, run_time=run_time)

    def change_equation(self, text_mobject, new_equation_str, run_time):
        new_text_mobject = Tex(new_equation_str)
        new_text_mobject.shift(text_mobject.get_corner(DOWN + LEFT) - new_text_mobject.get_corner(DOWN + LEFT))
        return Transform(text_mobject, new_text_mobject, run_time=run_time)

    def factorize_with_exp_0(self, int_array):
        all_primes_with_duplicates = []
        for index in range(len(int_array)):
            for prime in factorint(int_array[index]):
                all_primes_with_duplicates.append(prime)
        all_primes = []
        [all_primes.append(x) for x in all_primes_with_duplicates if x not in all_primes]
        all_primes.sort()
        prime_factorization_strs = []
        for index in range(len(int_array)):
            prime_factorization_strs.append('')
            for prime in all_primes:
                exponent = 0
                if prime in factorint(int_array[index]).keys():
                    exponent = factorint(int_array[index])[prime]

                if exponent == 1:
                    prime_factorization_strs[index] += str(prime) + '\cdot '
                else:
                    prime_factorization_strs[index] += str(prime) + '^{' + str(
                        exponent) + '}\cdot '

            if prime_factorization_strs[index][len(prime_factorization_strs[index]) - 2:
            len(prime_factorization_strs[index]): 1] == 't ':
                prime_factorization_strs[index] = prime_factorization_strs[index][0:
                                                                                  len(prime_factorization_strs[
                                                                                          index]) - 6: 1]
        return prime_factorization_strs

    def factorize_with_boxed(self, int_array):
        all_primes_with_duplicates = []
        for index in range(len(int_array)):
            for prime in factorint(int_array[index]):
                all_primes_with_duplicates.append(prime)
        all_primes = []
        [all_primes.append(x) for x in all_primes_with_duplicates if x not in all_primes]
        all_primes.sort()
        max_exponent_of_prime = []
        for prime in all_primes:
            max_exp = 0
            max_exp_index = 0
            if prime in factorint(int_array[0]):
                max_exp = factorint(int_array[0])[prime]

            for index in range(len(int_array)):
                if prime in factorint(int_array[index]) and factorint(int_array[index])[prime] > max_exp:
                    max_exp = factorint(int_array[index])[prime]
                    max_exp_index = index
            max_exponent_of_prime.append(max_exp_index)

        print(max_exponent_of_prime)

        prime_factorization_boxed_strs = []
        for index in range(len(int_array)):
            prime_factorization_boxed_strs.append('')
            for prime_index in range(len(all_primes)):
                exponent = 0
                boxed_str = ''
                if all_primes[prime_index] in factorint(int_array[index]).keys():
                    exponent = factorint(int_array[index])[all_primes[prime_index]]

                if exponent == 1:
                    boxed_str = str(all_primes[prime_index])
                else:
                    boxed_str = str(all_primes[prime_index]) + '^{' + str(exponent) + '}'
                if index == max_exponent_of_prime[prime_index]:
                    prime_factorization_boxed_strs[index] += '\\boxed{' + boxed_str +'}\cdot '
                else:
                    prime_factorization_boxed_strs[index] += boxed_str + '\cdot '

            if prime_factorization_boxed_strs[index][len(prime_factorization_boxed_strs[index]) - 2:
                                                     len(prime_factorization_boxed_strs[index]): 1] == 't ':
                prime_factorization_boxed_strs[index] = prime_factorization_boxed_strs[index][0:
                                                        len(prime_factorization_boxed_strs[index]) - 6: 1]
        return prime_factorization_boxed_strs

    def construct(self):
        lcm_inputs = [60, 73, 51]
        lcm_text = Tex('$\\text{lcm}($')
        lcm_input_text_str = '$'
        for int_to_be_factored in lcm_inputs:
            lcm_input_text_str = lcm_input_text_str + str(int_to_be_factored) + ', '
        lcm_input_text_str = lcm_input_text_str[0: len(lcm_input_text_str) - 2: 1] + '$'

        buff = 0.05
        lcm_input_text = Tex(lcm_input_text_str + '$)$')
        lcm_input_text.next_to(lcm_text, RIGHT, buff=buff)

        full_code = Tex('$\\text{lcm}($' + lcm_input_text_str + '$)$')

        # self.add(full_code)
        # helps center the expression

        def update_text(obj):
            obj.next_to(lcm_text, RIGHT, buff=buff)

        lcm_input_text.add_updater(update_text)
        initial_left_shift_factor = lcm_text.get_corner(direction=DOWN + LEFT)[0] - \
                                    full_code.get_corner(direction=DOWN + LEFT)[0]
        lcm_text.shift(LEFT * initial_left_shift_factor)
        # self.play(LCMExplanation.moveTo(self, lcm_text, LEFT))

        vertical_lcm_inputs_str = ''
        for inp in lcm_inputs:
            vertical_lcm_inputs_str = vertical_lcm_inputs_str + str(inp) + ' \\\\'

        lcm_text_final_location = 2.5 * LEFT + 2 * DOWN
        vertical_lcm_inputs = Tex(vertical_lcm_inputs_str)
        vertical_lcm_inputs.shift(vertical_lcm_inputs.get_corner(DOWN + RIGHT) -
                                  full_code.get_corner(UP + RIGHT) +
                                  UP)
        prime_factorization_vertical_str = '\\begin{tabular}{l} '

        for input in lcm_inputs:
            prime_factorization_vertical_str += '$' + str(input) + '=' + \
                                                str(LCMExplanation.prime_factorization_in_tex(self, input)) + \
                                                '$\\\\ '
        prime_factorization_vertical_str = prime_factorization_vertical_str[
                                           0: len(prime_factorization_vertical_str) - 3: 1] + '\\end{tabular}'
        prime_fac_exp0 = '\\begin{tabular}{l} '
        primefact_exp0_strs = LCMExplanation.factorize_with_exp_0(self, lcm_inputs)

        for index in range(len(lcm_inputs)):
            prime_fac_exp0 += '$' + str(lcm_inputs[index]) + '=' + str(primefact_exp0_strs[index]) + '$\\\\ '
        prime_fac_exp0 = prime_fac_exp0[0: len(prime_fac_exp0) - 3: 1] + '\\end{tabular}'

        prime_fac_boxed = '\\begin{tabular}{l} '
        primefact_box_strs = LCMExplanation.factorize_with_boxed(self, lcm_inputs)

        for index in range(len(lcm_inputs)):
            prime_fac_boxed += '$' + str(lcm_inputs[index]) + '=' + str(primefact_box_strs[index]) + '$\\\\ '
        prime_fac_boxed = prime_fac_boxed[0: len(prime_fac_boxed) - 3: 1] + '\\end{tabular}'

        lcm = 1
        for num in lcm_inputs:
            lcm = lcm * num // math.gcd(lcm, num)

        full_code.shift(lcm_text_final_location)

        lcm_str = '$\\text{lcm}($' + lcm_input_text_str + '$)=' + LCMExplanation.prime_factorization_in_tex(self, lcm) +\
                  '$'

        # prime_factorization_vertical = Tex(prime_factorization_vertical_str)
        # prime_factorization_vertical.shift(vertical_lcm_inputs.get_corner(DOWN + LEFT) -
        #                                   prime_factorization_vertical.get_corner(DOWN + LEFT))

        # self.add(Tex('$73=73$').shift(3 * UP + 3 * LEFT))
        self.play(FadeIn(lcm_text), FadeIn(lcm_input_text))
        self.wait()
        self.play(LCMExplanation.moveTo(self, lcm_text, lcm_text_final_location, 2),
                  TransformFromCopy(lcm_input_text, vertical_lcm_inputs, run_time=2))
        self.wait()
        self.play(LCMExplanation.change_equation(self, vertical_lcm_inputs, prime_factorization_vertical_str, 1))
        # self.play(Transform(vertical_lcm_inputs, prime_factorization_vertical))
        self.play(LCMExplanation.change_equation(self, vertical_lcm_inputs, prime_fac_exp0, 1))
        self.play(LCMExplanation.change_equation(self, vertical_lcm_inputs, prime_fac_boxed, 1))
        # self.play(LCMExplanation.change_equation(self, lcm_text, '$=$', 1))
        self.add(full_code)
        self.remove(lcm_text, lcm_input_text)
        self.play(LCMExplanation.change_equation(self, full_code, lcm_str, 1))

        
class LCMExplanation2(Scene):
    def prime_factorization_in_tex(self, num_to_be_factored):
        tex_strs = []
        for prime in factorint(num_to_be_factored):
            exponent = factorint(num_to_be_factored)[prime]
            tex_strs.append(str(prime))
            if exponent > 1:
                tex_strs.append('^{' + str(factorint(num_to_be_factored)[prime]) + '}')
            tex_strs.append('\cdot ')

        if tex_strs[len(tex_strs) - 1] == '\cdot ':
            tex_strs.pop(len(tex_strs) - 1)

        return tex_strs

    def moveTo(self, mobject, move_to_location, run_time):
        mobject.generate_target()
        mobject.target.shift(move_to_location)
        return MoveToTarget(mobject, run_time=run_time)

    def change_equation(self, text_mobject, new_equation_str, run_time):
        new_text_mobject = Tex(new_equation_str)
        new_text_mobject.shift(text_mobject.get_corner(DOWN + LEFT) - new_text_mobject.get_corner(DOWN + LEFT))
        return Transform(text_mobject, new_text_mobject, run_time=run_time)

    def factorize_array(self, int_array):
        tex_strs = []
        for num in int_array:
            tex_strs.append(str(num))
            tex_strs.append('&=')
            for string in LCMExplanation2.prime_factorization_in_tex(self, num):
                tex_strs.append(string)
            tex_strs.append('\\\\')
        tex_strs.pop()
        return tex_strs

    def factorize_with_exp_0(self, int_array):
        tex_strs = []

        all_primes_with_duplicates = []
        for index in range(len(int_array)):
            for prime in factorint(int_array[index]):
                all_primes_with_duplicates.append(prime)
        all_primes = []
        [all_primes.append(x) for x in all_primes_with_duplicates if x not in all_primes]
        all_primes.sort()
        for index in range(len(int_array)):
            tex_strs.append(str(int_array[index]))
            tex_strs.append('&=')
            for prime in all_primes:
                exponent = 0
                if prime in factorint(int_array[index]).keys():
                    exponent = factorint(int_array[index])[prime]
                tex_strs.append(str(prime))

                if exponent != 1:
                    tex_strs.append('^{' + str(exponent) + '}')

                tex_strs.append('\cdot ')

            if tex_strs[len(tex_strs) - 1] == '\cdot ':
                tex_strs.pop(len(tex_strs) - 1)
            tex_strs.append('\\\\')
        tex_strs.pop()
        return tex_strs

    def factorize_with_boxed(self, int_array):
        tex_strs = []

        all_primes_with_duplicates = []
        for index in range(len(int_array)):
            for prime in factorint(int_array[index]):
                all_primes_with_duplicates.append(prime)
        all_primes = []
        [all_primes.append(x) for x in all_primes_with_duplicates if x not in all_primes]
        all_primes.sort()
        max_exponent_of_prime = []
        for prime in all_primes:
            max_exp = 0
            max_exp_index = 0
            if prime in factorint(int_array[0]):
                max_exp = factorint(int_array[0])[prime]

            for index in range(len(int_array)):
                if prime in factorint(int_array[index]) and factorint(int_array[index])[prime] > max_exp:
                    max_exp = factorint(int_array[index])[prime]
                    max_exp_index = index
            max_exponent_of_prime.append(max_exp_index)

        for index in range(len(int_array)):
            tex_strs.append(str(int_array[index]))
            tex_strs.append('&=')
            for prime_index in range(len(all_primes)):
                exponent = 0
                if all_primes[prime_index] in factorint(int_array[index]).keys():
                    exponent = factorint(int_array[index])[all_primes[prime_index]]
                tex_strs.append(str(all_primes[prime_index]))

                if index == max_exponent_of_prime[prime_index]:
                    tex_strs.append('^{\color{red}' + str(exponent) + '}')
                elif exponent != 1:
                    tex_strs.append('^{' + str(exponent) + '}')

                if prime_index != len(all_primes) - 1:
                    tex_strs.append('\cdot ')

            tex_strs.append('\\\\')

        return tex_strs

    def locations_of_lcm_inputs(self, general_array):
        input_index_array = [0]
        while '&=' in general_array[input_index_array[-1] + 1: len(general_array) - 1: 1]:
            input_index_array.append(general_array.index('&=', input_index_array[-1] + 1, len(general_array) - 1))
        input_index_array.pop(0)
        for index in range(len(input_index_array)):
            input_index_array[index] += -1

        return input_index_array

    def construct(self):
        lcm_inputs = [60, 73, 51]

        lcm = 1
        for num in lcm_inputs:
            lcm = lcm * num // math.gcd(lcm, num)

        lcm_expression_starr = ['\\text{lcm}(']
        for input in lcm_inputs:
            lcm_expression_starr.append(str(input))
            lcm_expression_starr.append(',')
        lcm_expression_starr[len(lcm_expression_starr) - 1] = ')'

        lcm_expression = MathTex(*lcm_expression_starr)

        factored_inputs_starr = LCMExplanation2.factorize_array(self, lcm_inputs)
        factored_inputs = MathTex(*factored_inputs_starr)

        factored_exp_0_starr = LCMExplanation2.factorize_with_exp_0(self, lcm_inputs)
        factored_exp_0 = MathTex(*factored_exp_0_starr)

        factored_colored_exp_starr = LCMExplanation2.factorize_with_boxed(self, lcm_inputs)
        factored_colored_exp = MathTex(*factored_colored_exp_starr)

        indices_of_factored_lcm_inputs = LCMExplanation2.locations_of_lcm_inputs(self, factored_inputs_starr)
        indices_of_lcm_inputs_colored_exp = LCMExplanation2.locations_of_lcm_inputs(self, factored_colored_exp_starr)

        lcm_closed_paren_index = 2*len(lcm_inputs)

        lcm_expression_final_location = factored_colored_exp[indices_of_lcm_inputs_colored_exp[-1]].get_corner(DOWN+RIGHT) + \
                                        DOWN - lcm_expression[lcm_closed_paren_index].get_corner(UP + RIGHT)

        factored_inputs.shift(factored_colored_exp.get_corner(UP+LEFT) -
                              factored_inputs.get_corner(UP+LEFT))
        factored_exp_0.shift(factored_colored_exp.get_corner(UP+LEFT) -
                              factored_exp_0.get_corner(UP+LEFT))

        input_copies = []
        for index in range(len(lcm_inputs)):
            input_copies.append(lcm_expression[1+2*index].copy())

        move_copies = []
        for index in range(len(lcm_inputs)):
            move_copies.append(input_copies[index].animate.move_to(factored_inputs[indices_of_factored_lcm_inputs[index]].get_center()))

        moves_to_final_locations_1 = []
        previous_index_1 = 0
        for index in range(len(factored_inputs)):
            print(previous_index_1)
            next_index = factored_exp_0_starr.index(factored_inputs_starr[index], previous_index_1, len(factored_exp_0_starr)-1)
            move = factored_inputs[index].animate.move_to(factored_exp_0[next_index].get_center())
            previous_index_1 = next_index
            moves_to_final_locations_1.append(move)

        moves_to_final_locations_2 = []
        previous_index_2 = 0
        print(factored_exp_0_starr)
        print()
        print(factored_colored_exp_starr)
        for index in range(len(factored_exp_0)):
            if '^' not in factored_exp_0_starr[index]:
                next_index_2 = factored_colored_exp_starr.index(factored_exp_0_starr[index], previous_index_2,
                                                    len(factored_colored_exp_starr) - 1)
            elif factored_exp_0_starr[index][2: -1: 1] == '0':
                next_index_2 = factored_colored_exp_starr.index(factored_exp_0_starr[index], previous_index_2,
                                                                len(factored_colored_exp_starr) - 1)
            else:
                string_to_find = '^{\\color{red}' + factored_exp_0_starr[index][2: -1: 1] + '}'
                next_index_2 = factored_colored_exp_starr.index(string_to_find, previous_index_2,
                                                                len(factored_colored_exp_starr) - 1)
            move = factored_exp_0[index].animate.move_to(factored_colored_exp[next_index_2].get_center())
            previous_index_2 = next_index_2
            moves_to_final_locations_2.append(move)

        lcm_expression_starr.append('=')

        for string in LCMExplanation2.factorize_with_boxed(self, [lcm])[2: -1]:
            lcm_expression_starr.append(string)

        lcm_expression_final = MathTex(*lcm_expression_starr)
        lcm_expression_final.shift(lcm_expression[0].get_center() -
                                   lcm_expression_final[0].get_center() +
                                   lcm_expression_final_location)
        lcm_expression_final.set_color_by_tex('\\color{red}', RED)

        # self.play(*initial_lcm_fadeIn)
        self.play(FadeIn(lcm_expression))
        self.add(*input_copies)
        self.play(lcm_expression.animate.move_to(lcm_expression_final_location), *move_copies)
        self.play(FadeIn(factored_inputs))
        self.remove(*input_copies)
        self.play(*moves_to_final_locations_1)
        self.wait()
        self.play(FadeIn(factored_exp_0))
        self.remove(factored_inputs)
        self.play(*moves_to_final_locations_2)
        self.wait()
        self.play(FadeIn(factored_colored_exp))
        self.play(factored_colored_exp.animate.set_color_by_tex('\\color{red}', RED))
        self.play(FadeIn(lcm_expression_final))
        self.remove(lcm_expression)
        
        
class Logo(Scene):
    def construct(self):
        channel_name = MathTex("\\text{The}", " ")
        initial_circ = Circle(radius=0.037, fill_color=BLUE, fill_opacity=1, stroke_color=BLUE,
                              stroke_width=4)

        MAX_RADIUS = 1

        current_radius = 0.037 * 2

        circ_list = [initial_circ]

        counter = 1
        while current_radius < MAX_RADIUS:
            new_circ = Circle(radius=current_radius, stroke_width=4, stroke_color=BLUE,
                              stroke_opacity=math.pow(math.e, -6 * current_radius * current_radius))
            counter += 1
            current_radius += 0.037
            circ_list.append(new_circ)

        circles = VGroup(initial_circ, *circ_list)
        list = []
        integral = MathTex('\int')
        angle = TAU

        n = 5
        run_time = 2
        for index in range(n):
            list.append(MathTex('\int').rotate(math.pi * index / n).scale(3))

        integrals = VGroup(*list)
        integral_fade_ins = []

        order_of_appearence = [1, 2, 3, 4]
        for index in range(1, n):
            def update_sector(mob, alpha):
                # start_angle = interpolate(0, angle * 180 / math.pi, alpha)
                mob.become(
                    mob.set_opacity(interpolate(0, 1, alpha))
                )

            integral_fade_ins.append(UpdateFromAlphaFunc(integrals[index], update_sector,
                                                         rate_func=rate_functions.ease_in_out_expo,
                                                         run_time=run_time))

        fade_in_group = AnimationGroup(*integral_fade_ins, lag_ratio=1)

        self.play(Write(integrals[0]), run_time=run_time)
        self.play(fade_in_group, Rotating(integrals, radians=4*TAU, run_time=4*run_time,
                                          rate_func=rate_functions.ease_in_out_expo),
                  FadeIn(circles, run_time=4*run_time))
        self.wait()
        
        
class LogoNew(Scene):
	def construct(self):
		num_integrals = 5

		rate_func = rate_functions.ease_in_out_cubic
		channel_name = MathTex(*[r'\text{The Geometers}']).shift(2.6 * DOWN)
		initial_circ = Circle(radius=0.037, fill_color=BLUE, fill_opacity=1, stroke_color=BLUE, stroke_width=4)

		circ_list = [initial_circ]
		for current_radius in np.arange(0.037 * 2, 1, 0.037):
			circ_list.append(
				Circle(
					radius=current_radius,
					stroke_width=4,
					stroke_color=BLUE,
					stroke_opacity=np.exp(-6 * current_radius ** 2)
				)
			)
		circles = VGroup(initial_circ, *circ_list)

		integral_list = [MathTex(r'\int').rotate(PI * index / num_integrals).scale(3) for index in range(num_integrals)]
		integrals = VGroup(*integral_list)

		delay = 0.75
		hacking_delay = Rotate(Square(1).set_opacity(0), run_time=delay)
		integral_fade_ins = [hacking_delay]

		def update_sector(mob, alpha):
			mob.become(mob.set_opacity(interpolate(0, 1, alpha)))

		run_time = 2
		for integral in integrals[1:]:
			integral_fade_ins.append(
				UpdateFromAlphaFunc(
					integral, update_sector, rate_func=rate_func, run_time=run_time - 2 * delay / (num_integrals - 1)
				)
			)

		integral_fade_ins.append(hacking_delay)
		fade_in_group = AnimationGroup(*integral_fade_ins, lag_ratio=1)

		self.wait()
		self.play(Write(integrals[0]), run_time=run_time)
		self.play(
			Rotate(integrals, angle=np.round(2 * np.pi, 5), run_time=4 * run_time, rate_func=rate_func),
			fade_in_group,
			Rotate(integrals[0], angle=np.round(2 * np.pi, 5), run_time=4 * run_time, rate_func=rate_func),
			FadeIn(circles, run_time=4 * run_time),
			Write(channel_name, run_time=4 * run_time)
		)
		self.wait()
		

class LCMExplanationNew(Scene):
    def prime_factorization_in_tex(self, num_to_be_factored):
        tex_strs = []
        for prime in factorint(num_to_be_factored):
            exponent = factorint(num_to_be_factored)[prime]
            tex_strs.append(str(prime))
            if exponent > 1:
                tex_strs.append('^{' + str(factorint(num_to_be_factored)[prime]) + '}')
            tex_strs.append('\cdot ')

        if tex_strs[len(tex_strs) - 1] == '\cdot ':
            tex_strs.pop(len(tex_strs) - 1)

        return tex_strs


    def prime_factorization_in_tex_wexp_1(self, num_to_be_factored):
        tex_strs = []
        for prime in factorint(num_to_be_factored):
            exponent = factorint(num_to_be_factored)[prime]
            tex_strs.append(str(prime))
            tex_strs.append('^{' + str(factorint(num_to_be_factored)[prime]) + '}')
            tex_strs.append('\cdot ')

        if tex_strs[len(tex_strs) - 1] == '\cdot ':
            tex_strs.pop(len(tex_strs) - 1)

        return tex_strs

    def moveTo(self, mobject, move_to_location, run_time):
        mobject.generate_target()
        mobject.target.shift(move_to_location)
        return MoveToTarget(mobject, run_time=run_time)

    def change_equation(self, text_mobject, new_equation_str, run_time):
        new_text_mobject = Tex(new_equation_str)
        new_text_mobject.shift(text_mobject.get_corner(DOWN + LEFT) - new_text_mobject.get_corner(DOWN + LEFT))
        return Transform(text_mobject, new_text_mobject, run_time=run_time)

    def factorize_array(self, int_array):
        tex_strs = []
        for num in int_array:
            tex_strs.append(str(num))
            tex_strs.append('&=')
            for string in LCMExplanation.prime_factorization_in_tex(self, num):
                tex_strs.append(string)
            tex_strs.append('\\\\')
        tex_strs.pop()
        return tex_strs

    def factorize_array_1(self, int_array):
        tex_strs = []
        for num in int_array:
            tex_strs.append(str(num))
            tex_strs.append('&=')
            for string in LCMExplanation.prime_factorization_in_tex_wexp_1(self, num):
                tex_strs.append(string)
            tex_strs.append('\\\\')
        tex_strs.pop()
        return tex_strs

    def factorize_with_exp_0(self, int_array):
        tex_strs = []

        all_primes_with_duplicates = []
        for index in range(len(int_array)):
            for prime in factorint(int_array[index]):
                all_primes_with_duplicates.append(prime)
        all_primes = []
        [all_primes.append(x) for x in all_primes_with_duplicates if x not in all_primes]
        all_primes.sort()
        for index in range(len(int_array)):
            tex_strs.append(str(int_array[index]))
            tex_strs.append('&=')
            for prime in all_primes:
                exponent = 0
                if prime in factorint(int_array[index]).keys():
                    exponent = factorint(int_array[index])[prime]
                tex_strs.append(str(prime))

                if exponent != 1:
                    tex_strs.append('^{' + str(exponent) + '}')

                tex_strs.append('\cdot ')

            if tex_strs[len(tex_strs) - 1] == '\cdot ':
                tex_strs.pop(len(tex_strs) - 1)
            tex_strs.append('\\\\')
        tex_strs.pop()
        return tex_strs

    def factorize_with_boxed(self, int_array):
        tex_strs = []

        all_primes_with_duplicates = []
        for index in range(len(int_array)):
            for prime in factorint(int_array[index]):
                all_primes_with_duplicates.append(prime)
        all_primes = []
        [all_primes.append(x) for x in all_primes_with_duplicates if x not in all_primes]
        all_primes.sort()
        max_exponent_of_prime = []
        for prime in all_primes:
            max_exp = 0
            max_exp_index = 0
            if prime in factorint(int_array[0]):
                max_exp = factorint(int_array[0])[prime]

            for index in range(len(int_array)):
                if prime in factorint(int_array[index]) and factorint(int_array[index])[prime] > max_exp:
                    max_exp = factorint(int_array[index])[prime]
                    max_exp_index = index
            max_exponent_of_prime.append(max_exp_index)

        for index in range(len(int_array)):
            tex_strs.append(str(int_array[index]))
            tex_strs.append('&=')
            for prime_index in range(len(all_primes)):
                exponent = 0
                if all_primes[prime_index] in factorint(int_array[index]).keys():
                    exponent = factorint(int_array[index])[all_primes[prime_index]]
                tex_strs.append(str(all_primes[prime_index]))

                if index == max_exponent_of_prime[prime_index]:
                    tex_strs.append('^{\color{red}' + str(exponent) + '}')
                # elif exponent != 1:
                #    tex_strs.append('^{' + str(exponent) + '}')
                else:
                    tex_strs.append('^{' + str(exponent) + '}')

                if prime_index != len(all_primes) - 1:
                    tex_strs.append('\cdot ')

            tex_strs.append('\\\\')

        return tex_strs

    def locations_of_lcm_inputs(self, general_array):
        input_index_array = [0]
        while '&=' in general_array[input_index_array[-1] + 1: len(general_array) - 1: 1]:
            input_index_array.append(general_array.index('&=', input_index_array[-1] + 1, len(general_array) - 1))
        input_index_array.pop(0)
        for index in range(len(input_index_array)):
            input_index_array[index] += -1
        # for item in input_index_array:
        #    item += -1

        # Use for each loop to update list

        return input_index_array

    def construct(self):
        # 101, 106, 77
        # 20, 32, 18, 56
        # 32, 7, 19, 82
        # 25, 85, 60, 76
        # 347, 441, 77, 217
        # 93, 401, 494
        # 455, 500, 340, 117
        lcm_inputs = [455, 500, 340, 117]
        if False:
            arr_length = random.randint(3, 4)
            for index in range(arr_length):
                lcm_inputs.append(random.randint(2, 500))

        lcm = 1
        for num in lcm_inputs:
            lcm = lcm * num // math.gcd(lcm, num)

        lcm_expression_starr = ['\\text{lcm}(']
        for input in lcm_inputs:
            lcm_expression_starr.append(str(input))
            lcm_expression_starr.append(',')
        lcm_expression_starr[len(lcm_expression_starr) - 1] = ')'

        lcm_expression = MathTex(*lcm_expression_starr)

        factored_inputs_starr = LCMExplanation.factorize_array(self, lcm_inputs)
        factored_inputs = MathTex(*factored_inputs_starr)

        factored_exp_1_starr = LCMExplanation.factorize_array_1(self, lcm_inputs)
        factored_exp_1 = MathTex(*factored_exp_1_starr)

        factored_exp_0_starr = LCMExplanation.factorize_with_exp_0(self, lcm_inputs)
        factored_exp_0 = MathTex(*factored_exp_0_starr)

        factored_colored_exp_starr = LCMExplanation.factorize_with_boxed(self, lcm_inputs)
        factored_colored_exp = MathTex(*factored_colored_exp_starr).shift(RIGHT)

        indices_of_factored_lcm_inputs = LCMExplanation.locations_of_lcm_inputs(self, factored_inputs_starr)
        indices_of_lcm_inputs_colored_exp = LCMExplanation.locations_of_lcm_inputs(self, factored_colored_exp_starr)

        lcm_closed_paren_index = 2 * len(lcm_inputs)

        lcm_expression_final_location = factored_colored_exp[indices_of_lcm_inputs_colored_exp[-1]].get_corner(
            DOWN + RIGHT) + \
                                        DOWN - lcm_expression[lcm_closed_paren_index].get_corner(UP + RIGHT)

        factored_inputs.shift(factored_colored_exp.get_corner(UP + LEFT) -
                              factored_inputs.get_corner(UP + LEFT))
        factored_exp_0.shift(factored_colored_exp.get_corner(UP + LEFT) -
                             factored_exp_0.get_corner(UP + LEFT))
        factored_exp_1.shift(factored_colored_exp.get_corner(UP + LEFT) -
                             factored_exp_1.get_corner(UP + LEFT))

        input_copies = []
        for index in range(len(lcm_inputs)):
            input_copies.append(lcm_expression[1 + 2 * index].copy())

        move_copies = []
        for index in range(len(lcm_inputs)):
            move_copies.append(input_copies[index].animate.move_to(
                factored_inputs[indices_of_factored_lcm_inputs[index]].get_center()))

        moves_to_final_locations_1 = []
        moves_to_final_locations_2 = []
        alt_moves_to_final_locations_1 = []

        previous_index_1 = 0
        alt_prev_index = 0
        for index in range(len(factored_inputs)):
            # print(previous_index_1)
            next_index = factored_exp_0_starr.index(factored_inputs_starr[index], previous_index_1,
                                                    len(factored_exp_0_starr))
            move = factored_inputs[index].animate.move_to(factored_exp_0[next_index].get_center())
            previous_index_1 = next_index
            moves_to_final_locations_1.append(move)

        for index in range(len(factored_inputs)):
            # print(previous_index_1)
            next_index = factored_exp_1_starr.index(factored_inputs_starr[index], alt_prev_index,
                                                    len(factored_exp_1_starr))
            move = factored_inputs[index].animate.move_to(factored_exp_1[next_index].get_center())
            alt_prev_index = next_index
            alt_moves_to_final_locations_1.append(move)

        previous_index_2 = 0
        print(factored_exp_0_starr)
        print()
        print(factored_colored_exp_starr)
        print()
        print(lcm_inputs)
        # next_index = 0
        # change factored_exp_1 to factored_exp_0 to reverse changes
        for index in range(len(factored_exp_1)):
            color_string = '^{\\color{red}' + factored_exp_1_starr[index][2: -1: 1] + '}'
            next_index_cand_1 = len(factored_colored_exp_starr) + 1
            next_index_cand_2 = len(factored_colored_exp_starr) + 1
            if color_string in factored_colored_exp_starr[previous_index_2: len(factored_colored_exp_starr)]:
                next_index_cand_1 = factored_colored_exp_starr.index(color_string, previous_index_2,
                                                                     len(factored_colored_exp_starr))
            if factored_exp_1_starr[index] in factored_colored_exp_starr[
                                              previous_index_2: len(factored_colored_exp_starr)]:
                next_index_cand_2 = factored_colored_exp_starr.index(factored_exp_1_starr[index], previous_index_2,
                                                                     len(factored_colored_exp_starr))
            next_index_2 = min(next_index_cand_1, next_index_cand_2)
            move = factored_exp_1[index].animate.move_to(factored_colored_exp[next_index_2].get_center())
            previous_index_2 = next_index_2
            moves_to_final_locations_2.append(move)

        lcm_expression_starr.append('=')

        for string in LCMExplanation.factorize_with_boxed(self, [lcm])[2: -1]:
            lcm_expression_starr.append(string)

        lcm_expression_final = MathTex(*lcm_expression_starr)
        lcm_expression_final.shift(lcm_expression[0].get_center() -
                                   lcm_expression_final[0].get_center() +
                                   lcm_expression_final_location)
        lcm_expression_final.set_color_by_tex('\\color{red}', RED)

        # self.play(*initial_lcm_fadeIn)
        self.play(FadeIn(lcm_expression))
        self.add(*input_copies)
        self.play(lcm_expression.animate.move_to(lcm_expression_final_location), *move_copies)
        self.play(FadeIn(factored_inputs))
        self.remove(*input_copies)
        self.play(*alt_moves_to_final_locations_1)
        self.wait()
        self.play(FadeIn(factored_exp_1))
        self.remove(factored_inputs)
        self.play(*moves_to_final_locations_2)
        self.wait()
        self.play(FadeIn(factored_colored_exp))
        self.play(factored_colored_exp.animate.set_color_by_tex('\\color{red}', RED))
        self.play(FadeIn(lcm_expression_final))
        self.remove(lcm_expression)


class CycleLengths2(Scene):
    def moveTo(self, mobject, move_to_location, run_time):
        mobject.generate_target()
        mobject.target.shift(move_to_location-mobject.get_center())
        return MoveToTarget(mobject, run_time=run_time)

    def construct(self):
        equations = MathTex(
            '\\text{lcm}(', 'c_1', ',', 'c_2', ',', '\\dots', ',',  'c_m', ') &=',
            '1000',
            '\color{black} = 2^3 \cdot 5^3\\\\',
            'c_1 + c_2 + ... + c_m &= n'
        )
        # c_1 --> 1
        # c_2 --> 3
        # ... --> 5
        # c_m --> 7

        the_rest = [index for index in range(len(equations))]
        the_rest.pop(7)
        the_rest.pop(5)
        the_rest.pop(3)
        the_rest.pop(1)

        vertical_eqs = MathTex(
            'c_1', '&=', '2', '^{a_1}', '5', '^{b_1}\\\\',
            'c_2', '&=', '2', '^{a_2}', '5', '^{b_2}\\\\',
            '\\vdots \\\\',
            'c_m', '&=', '2', '^{a_m}', '5', '^{b_m}',
        ).shift(UP)

        # c_1 --> 0
        # c_2 --> 6
        # \vdots --> 12
        # c_m --> 13
        # = --> 14

        vertical_eqs_cj = MathTex(
            'c_1', '&=', '2', '^{a_1}', '5', '^{b_1}\\\\',
            'c_2', '&=', '2', '^{a_2}', '5', '^{b_2}\\\\',
            '\\vdots \\\\',
            'c_j', '&=', '2', '^{a_j}', '5', '^{b_j}\\\\',
            '\\vdots\\\\',
            'c_m', '&=', '2', '^{a_m}', '5', '^{b_m}',
        ).shift(UP)

        # 12, 19
        vertical_eqs_cj[12].set_opacity(0)
        vertical_eqs_cj[19].set_opacity(0)


        lcm_equation = MathTex(
            '\\text{lcm}(', 'c_1', ',', 'c_2', ',', '\\dots', ',',  'c_m', ')', '&=', '1000'
        )
        lcm_equation_final = MathTex(
            '\\text{lcm}(', 'c_1', ',', 'c_2', ',', '\\dots', ',',  'c_m', ')', '&=', '2', '^{3}',
            '\cdot', '5', '^{3}'
        )

        # = --> 9

        shift_vector = vertical_eqs[14].get_center()+DOWN-lcm_equation[9].get_center()
        shift_vector_final = vertical_eqs[14].get_center()+DOWN-lcm_equation_final[9].get_center()

        lcm_equation.shift(shift_vector)
        lcm_equation_final.shift(shift_vector_final)

        equations[5].save_state()
        #vertical_eqs[12].shift(equations[5].get_center()-vertical_eqs[12].get_center())
        vertical_eqs[12].set_opacity(0)

        def ellipses_mover(mob, alpha):
            mob.restore()
            desti = vertical_eqs[12].get_center()+[vertical_eqs[0].get_x() - vertical_eqs[12].get_x(), 0, 0]
            start = equations[5].get_center()
            mob.become(
                MathTex('\dots').shift(
                    start+(desti-start)*interpolate(0, 1, alpha)
                ).rotate(
                    interpolate(0,
                                np.round(PI/2, 5),
                                alpha))
            )

        run_time = 2

        c_moves = [UpdateFromAlphaFunc(equations[5], ellipses_mover, run_time=run_time),
                   equations[1].animate(run_time=run_time).move_to(vertical_eqs[0].get_center()),
                   equations[3].animate(run_time=run_time).move_to(vertical_eqs[6].get_center()),
                   equations[7].animate(run_time=run_time).move_to(vertical_eqs[13].get_center())]

        fade_out_equations = [FadeOut(equations[index], run_time=run_time/2) for index in the_rest]

        cj_moves = [CycleLengths2.moveTo(self, vertical_eqs[index], vertical_eqs_cj[index].get_center(), run_time) for index in range(12)]

        dots_copy = MathTex('\dots')
        dots_copy.rotate(np.round(PI / 2, 5))
        dots_copy.move_to(vertical_eqs[12].get_center()+[vertical_eqs[0].get_x() - vertical_eqs[12].get_x(), 0, 0])

        dots_copy_2 = MathTex('\dots')
        dots_copy_2.rotate(np.round(PI / 2, 5))
        dots_copy_2.move_to(vertical_eqs[12].get_center() + [vertical_eqs[0].get_x() - vertical_eqs[12].get_x(), 0, 0])

        cj_moves.append(CycleLengths2.moveTo(self, dots_copy, [dots_copy.get_x(), vertical_eqs_cj[19].get_y(), 0], run_time))
        cj_moves.append(CycleLengths2.moveTo(self, dots_copy_2, [dots_copy_2.get_x(), vertical_eqs_cj[12].get_y(), 0], run_time))
        for index in range(13, 19):
            cj_moves.append(vertical_eqs[index].animate(run_time=run_time).move_to(vertical_eqs_cj[index+7].get_center()))

        shift_vec = -vertical_eqs[13].get_center()+vertical_eqs_cj[20].get_center()
        cj_moves.append(lcm_equation_final.animate(run_time=run_time).shift(shift_vec))

        self.play(FadeIn(*equations))
        self.play(*c_moves, *fade_out_equations)
        self.wait()
        self.play(FadeIn(vertical_eqs))
        self.add(dots_copy, dots_copy_2)
        self.remove(equations[1], equations[3], equations[5], equations[7])
        self.wait()
        self.play(Write(lcm_equation, run_time=run_time))
        self.wait()
        self.play(FadeOut(lcm_equation[-1]))
        self.wait()
        self.play(FadeIn(lcm_equation_final))
        self.remove(*[lcm_equation[index] for index in range(len(lcm_equation)) if index != -1])
        self.wait()
        self.play(vertical_eqs.animate.set_color_by_tex('^{a', RED))
        self.wait()
        self.play(vertical_eqs.animate.set_color_by_tex('^{a', WHITE))
        self.wait()
        self.play(*cj_moves)
        self.wait()
        self.play(FadeIn(vertical_eqs_cj))
        self.remove(vertical_eqs)
        self.wait()

	
	
class Preamble(Scene):
    def scale_from_point(self, group, point, factor):
        moves = []
        for obj in group:
            dist = linalg.norm(obj.get_center() - point.get_center())
            if dist != 0:
                direction = (obj.get_center() - point.get_center()) / dist
                moves.append(obj.animate.move_to(point.get_center() + direction * dist * factor))

        return moves

    def construct(self):
        num_dots = 5
        spacing = 1
        dots = [Dot(LEFT * (num_dots - 1) / 2 * spacing + spacing * ind * RIGHT) for ind in range(num_dots)]
        labels = [Integer(num + 1).scale(0.5).next_to(dot, DOWN) for num, dot in enumerate(dots)]
        for dot, label in zip(dots, labels):
            label.add_updater(lambda l, dot=dot, label=label: l.next_to(dot, DOWN))

        self.play(FadeIn(*dots, *labels))

        curre = [num_dots - 1]

        def permute():
            anims = []
            loc = dots[(curre[0] + 1) % num_dots].get_center()
            run_time = 1
            new_dot = Dot(loc)
            copy = labels[curre[0]].next_to(dots[curre[0]], DOWN)
            self.remove(labels[curre[0]])
            lab = Integer(curre[0] + 1).scale(0.5).next_to(new_dot, DOWN)
            for index in range(num_dots):
                if index == curre[0]:
                    g1 = AnimationGroup(FadeOut(copy), FadeOut(dots[curre[0]]), run_time=run_time / 2)
                    g3 = AnimationGroup(FadeIn(new_dot, run_time=run_time / 2), FadeIn(lab, run_time=run_time / 2))
                    group = AnimationGroup(g1, g3, lag_ratio=1)
                    anims.append(group)
                else:
                    anims.append(dots[index].animate(run_time=run_time).move_to(dots[(index + 1) % num_dots]))
            self.play(*anims)
            dots[curre[0] % num_dots].move_to(loc)
            self.add(labels[curre[0]], dots[curre[0]])
            self.remove(lab, new_dot)
            curre[0] = (curre[0] - 1) % num_dots

        for ind in range(5):
            permute()

        #self.play(Transform(labels[1], MathTex('H').scale(0.5).move_to(labels[1].get_center())))

        shift_vec = 0.25*LEFT

        moves = []
        moves.append(dots[-1].animate.move_to(dots[3].get_center()+shift_vec))
        moves.append(dots[3].animate.shift(0.5 * UP + 0.5 * LEFT+shift_vec))
        moves.append(dots[2].animate.shift(RIGHT + 0.5 * DOWN + 0.5 * LEFT+shift_vec))
        moves.append(dots[1].animate.shift(RIGHT * 1.1 + UP * 1.1 + shift_vec))
        moves.append(dots[0].animate.shift(RIGHT * 2.1 + DOWN * 1.1+shift_vec))

        self.play(*moves, FadeOut(*labels))

        Fg = Arrow(start=[0, 0, 0], end=1.5*DOWN, tip_shape=CustomArrowTip, buff=0, stroke_width=2)

        grp = VGroup(*dots)
        self.play(*Preamble.scale_from_point(self, grp, dots[-1], 1.5))

        buff1 = 0.25

        bond1 = make_arrow_between(dots[4], dots[3], buff=buff1)
        bond2 = make_arrow_between(dots[4], dots[2], buff=buff1)
        bond3 = make_arrow_between(dots[2], dots[3], buff=buff1).shift(LEFT * 0.05)
        bond4 = make_arrow_between(dots[2], dots[3], buff=buff1).shift(RIGHT * 0.05)
        bond5 = make_arrow_between(dots[1], dots[3], buff=buff1)
        bond6 = make_arrow_between(dots[0], dots[2], buff=buff1)

        c4 = MathTex('C').scale(0.5).move_to(dots[4].get_center())
        c3 = MathTex('C').scale(0.5).move_to(dots[3].get_center())
        c2 = MathTex('C').scale(0.5).move_to(dots[2].get_center())
        c1 = MathTex('H').scale(0.5).move_to(dots[1].get_center())
        c0 = MathTex('H').scale(0.5).move_to(dots[0].get_center())

        electron1 = Dot().scale(0.3).move_to(dots[4].get_center()+0.1*(RIGHT+UP)+RIGHT*0.05)
        electron2 = Dot().scale(0.3).move_to(dots[4].get_center()+0.1*(RIGHT+DOWN)+RIGHT*0.05)

        self.play(Transform(dots[4], c4),
                  Transform(dots[3], c3),
                  Transform(dots[2], c2),
                  Transform(dots[1], c1),
                  Transform(dots[0], c0),
                  FadeIn(bond1, bond2, bond3, bond4, bond5, bond6),
                  FadeIn(electron1, electron2))

        line1 = make_arrow_between(dots[4], dots[3], 0)
        line2 = make_arrow_between(dots[4], dots[2], 0)
        line6 = make_arrow_between(dots[0], dots[2], 0)
        line5 = make_arrow_between(dots[1], dots[3], 0)

        new_dots = [Dot(dots[ind].get_center()) for ind in range(num_dots)]

        self.play(Transform(bond1, line1),
                  Transform(bond2, line2),
                  Transform(bond5, line5),
                  Transform(bond6, line6),
                  Transform(dots[4], new_dots[4]),
                  Transform(dots[3], new_dots[3]),
                  Transform(dots[2], new_dots[2]),
                  Transform(dots[1], new_dots[1]),
                  Transform(dots[0], new_dots[0]),
                  FadeOut(bond3, bond4, electron1, electron2))

        arrow_add_sticky_updater(bond5, dots[1], dots[3])
        arrow_add_sticky_updater(bond6, dots[0], dots[2])

        arrow_add_sticky_updater(bond1, dots[4], dots[3])

        self.play(dots[0].animate.move_to(dots[4].get_center()+1.5*LEFT),
                  dots[1].animate.move_to(dots[4].get_center()+1.5*LEFT))

        self.play(FadeIn(Fg))
        self.play(grp.animate(rate_func=rate_functions.ease_in_quad).move_to(DOWN*5),
                  Fg.animate(rate_func=rate_functions.ease_in_quad).shift(DOWN*5),
                  bond2.animate(rate_func=rate_functions.ease_in_quad).shift(DOWN*5))
        self.wait()

        poly = RegularPolygon(10, stroke_color=WHITE).scale(2)

        self.play(FadeIn(poly))
        self.play(Rotate(poly, np.round(TAU/10, 13)))
        self.play(Rotate(poly, np.round(TAU/5, 13)))
        self.play(Rotate(poly, axis=UP))

        abstract = MathTex('\\text{Aut}(C_p^n)=\{f: C_p^n\\to C_p^n\mid f\\text{ is an isomorphism}\}')

        self.play(Transform(poly, abstract))
        self.play(FadeOut(poly))

        orb = MathTex('\\text{Orbit-Stabilizer Theorem}').shift(UP*0.25)
        below_text_1 = MathTex('\\text{Nope, deeper than this}').scale(0.5).shift(DOWN*0.40)

        self.play(FadeIn(orb))
        self.play(FadeIn(below_text_1))
        self.wait()

        classification = MathTex('\\text{The Classification of Finite Simple Groups}').shift(UP*0.25)
        below_text_2 = MathTex('\\text{No, not }that\\text{ deep}').scale(0.5).shift(DOWN*0.4)

        unsolvability = MathTex('\\text{Unsolvability of the Quintic}').shift(UP*0.25)
        below_text_3 = MathTex('\\text{Perfect!}').scale(0.5).shift(DOWN*0.4)

        self.play(FadeOut(below_text_1), Transform(orb, classification))
        self.play(FadeIn(below_text_2))
        self.wait()

        self.play(FadeOut(below_text_2), Transform(orb, unsolvability))
        self.play(FadeIn(below_text_3))
        self.wait()

        self.play(FadeOut(orb, below_text_3))


	class IdentityIn5(Scene):
    num_permutes = 0

    def permute(self, dots, labels, permutations):
        transforms = []
        create_arrows = []
        destroy_arrows = []
        fade_out_labels = [FadeOut(label) for label in labels]

        for permutation in permutations:
            for index in range(len(permutation) - 1):
                transforms.append(
                    Transform(
                        dots[permutation[index]], dots[permutation[index + 1]]
                    )
                )
                arrow = make_arrow_between(dots[permutation[index]], dots[permutation[index + 1]])
                create_arrows.append(FadeIn(arrow))
                destroy_arrows.append(FadeOut(arrow))

        self.play(*create_arrows)
        self.play(*fade_out_labels)
        self.play(*transforms, *destroy_arrows, run_time=2)
        self.num_permutes += 1
        fade_in_labels = [FadeIn(label) for label in labels]
        for label, dot in zip(labels, dots):
            label.move_to(dot.get_center() * 1.1)

        self.play(*fade_in_labels)

    def construct(self):
        circle = Circle(radius=3, color=BLACK)
        self.add(circle)

        num_points = 20

        dots = []
        labels = []

        for c, angle in enumerate(np.linspace(0, TAU, num_points, endpoint=False)):
            point = circle.point_at_angle(angle)
            dot = Dot(point=point)
            dots.append(dot)
            label = Integer(number=c + 1).scale(0.5).move_to(dot.get_center() * 1.1)
            labels.append(label)
            self.add(dot, label)

        permutations = [
            [0, 4, 8, 12, 16, 0],
            [1, 5, 9, 13, 17, 1],
            [2, 6, 10, 14, 18, 2],
            [3, 7, 11, 15, 19, 3],
        ]

        counter = Integer(0).scale(0.5).move_to(RIGHT * 4 + UP * 3.5)
        counter.add_updater(lambda i: i.set_value(self.num_permutes))
        self.add(counter)

        for _ in range(5):
            self.permute(dots, labels, permutations)
            self.wait()


def make_arrow_between(dot1, dot2, buff):
    arrow = Line(
        dot1.get_center(),
        dot2.get_center(),
        stroke_width=3,
        buff=buff
    )

    return arrow


def arrow_add_sticky_updater(arrow, dot1, dot2):
    arrow.add_updater(
        lambda arw: arw.put_start_and_end_on(
            Line(start=dot1.get_center(), end=dot2.get_center()).scale(0.9).get_start(),
            Line(start=dot1.get_center(), end=dot2.get_center()).scale(0.9).get_end()
        )
    )


def make_sticky_arrow_between(dot1, dot2):
    arrow = make_arrow_between(dot1, dot2)
    arrow_add_sticky_updater(arrow, dot1, dot2)

    return arrow
