
import math
from abc import ABC

from manim import *


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
        for index in range(n):
            list.append(MathTex('\int').rotate(2 * math.pi * index / n).scale(3))

        integrals = VGroup(*list)
        inte = list[1]
        inte.save_state()

        def update_sector(mob, alpha):
            mob.restore()
            start_angle = interpolate(0, angle*180/math.pi, alpha)
            mob.become(
                inte.rotate(start_angle*DEGREES).set_opacity(interpolate(0, 1, alpha))
            )
        run_time = 2
        self.add(integrals[0])

        group1 = AnimationGroup(*[UpdateFromAlphaFunc(inte, update_sector,
                                rate_func=rate_functions.ease_in_out_expo, run_time=run_time),
            Rotating(integrals[0], radians=angle, rate_func=rate_functions.ease_in_out_expo, run_time=run_time)], lag_ratio=0)
        #self.play(group1)
        # self.add(Line(LEFT, RIGHT).set_color(BLUE))
        #self.wait()
        # self.add(line)
        # self.play(Rotating(line), line.animate.set_color(BLUE))
        self.add(*list, *circles)
        #self.play(FadeIn(*list), FadeIn(*circ_list))
        #self.wait()

