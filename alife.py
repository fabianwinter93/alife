import pyglet
from pyglet import shapes
import pymunk
from pymunk import Vec2d
import numpy as np

import jax
from jax import tree_util
from jax import numpy as jnp
from pymunk.body import _BodyType

class Window(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def rnd_x(self):
        return np.random.randint(50, self.width - 50)
    
    def rnd_y(self):
        return np.random.randint(50, self.height - 50)
    
    def pos_in_bounds(self, x,y):
        if x > self.width or x < 0: 
            return False
        elif y > self.height or y < 0:
            return False
        else:
            return True
window = Window(4*300, 3*200)
window.set_fullscreen() 

batch = pyglet.graphics.Batch()
debug_batch = pyglet.graphics.Batch()
fpsdisplay = pyglet.window.FPSDisplay(window)

DEBUG_DRAW = False

@window.event
def on_draw():
    window.clear()
    batch.draw()
    if DEBUG_DRAW:
        debug_batch.draw()
    fpsdisplay.draw()

@window.event 
def on_key_press(symbol, modifier): 
    global DEBUG_DRAW
    # key "C" get press 
    if symbol == pyglet.window.key.D: 
          DEBUG_DRAW = not DEBUG_DRAW

WIDTH = window.width
HEIGHT = window.height

collision_types = {
    "boundaries": 0,
    "sight": 1,
    "creatures": 2,
    "food": 3
}

THREADS = 8

MAX_SIZE = 15
MIN_SIZE = 2

MAX_MASS = 100
MIN_MASS = 1

MAX_SIGHT = 200
MIN_SIGHT = 30

MAX_HP = 30
MIN_HP = 10

def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))

def layernorm(x):
    m = x.mean()
    s = x.std()
    return (x-m)/(s+1e-7)

def RADIUS_FROM_SIZE_FACTOR(size_factor):
    size = size_factor * (MAX_SIZE - MIN_SIZE) + MIN_SIZE
    return size

def MASS_FROM_SIZE_FACTOR(size_factor):
    mass = size_factor * (MAX_MASS - MIN_MASS) + MIN_MASS
    return mass
    
def HP_FROM_SIZE_FACTOR(size_factor):
    hp = size_factor * (MAX_HP - MIN_HP) + MIN_HP
    hp = np.clip(np.rint(hp), MIN_HP, MAX_HP)
    hp = int(hp)
    return hp

def add_creature_boundaries_collision(space):
    h = space.add_collision_handler(collision_types["creatures"], collision_types["boundaries"])

    def f(arbiter, space, data):
        assert len(arbiter.shapes) == 2
        _self = arbiter.shapes[0]
        other = arbiter.shapes[1]

        _self.body.owner.take_damage(10)
        

        return True
        

    h.post_solve = f    

def add_creature_creature_collision(space):
    h = space.add_collision_handler(collision_types["creatures"], collision_types["creatures"])

    def f(arbiter, space, data):
        assert len(arbiter.shapes) == 2
        _self = arbiter.shapes[0]
        other = arbiter.shapes[1]

        if arbiter.is_first_contact:

            if other.body.mass > _self.body.mass:
                _self.body.owner.take_damage(1)
            else:
                other.body.owner.take_damage(1)
        

        return True
        

    h.post_solve = f    

def add_creature_food_collision(space):
    h = space.add_collision_handler(collision_types["creatures"], collision_types["food"])

    def f(arbiter, space, data):
        has_eaten = arbiter.shapes[0].body.owner.maybe_eat_food(1)
        if has_eaten:
            arbiter.shapes[1].body.owner.value -= 1

    h.post_solve = f    

def add_sight_creature_collision(space):
    h = space.add_collision_handler(collision_types["sight"], collision_types["creatures"])

    def f(arbiter, space, data):
        assert len(arbiter.shapes) == 2
        _self = arbiter.shapes[0]
        other = arbiter.shapes[1]

        other_pos = other.body.position
        dist = _self.body.position - other_pos

        _self.body.owner.creatures_in_sight.append([collision_types["creatures"], dist])

        return True

    h.pre_solve = f

def add_sight_food_collision(space):
    h = space.add_collision_handler(collision_types["sight"], collision_types["food"])

    def f(arbiter, space, data):
        _self = arbiter.shapes[0]
        other = arbiter.shapes[1]

        other_pos = other.body.position
        dist = _self.body.position - other_pos

        _self.body.owner.creatures_in_sight.append([collision_types["food"], dist])

        return True

    h.pre_solve = f

class Params:
    def __init__(self):
        self.in_dim = None
        self.state_dim = None
        self.out_dim = None

        self.wi = None
        self.bi = None

        self.wf = None
        self.bf = None
        
        self.wc = None
        self.bc = None

        self.wo = None
        self.bo = None

        self.wy = None
        self.by = None

        self.params = [self.wi, self.bi, self.wf, self.bf, self.wc, self.bc, self.wo, self.bo, self.wy, self.by]
        
    @classmethod
    def new(cls, in_dim, state_dim, out_dim):
        p = Params()
        p.in_dim = in_dim
        p.state_dim = state_dim
        p.out_dim = out_dim

        p.wi = np.random.normal(size=(in_dim, state_dim))
        p.bi = np.zeros((state_dim,))

        p.wf = np.random.normal(size=(in_dim, state_dim))
        p.bf = np.zeros((state_dim,))

        p.wc = np.random.normal(size=(in_dim, state_dim))
        p.bc = np.zeros((state_dim,))

        p.wo = np.random.normal(size=(in_dim, state_dim))
        p.bo = np.zeros((state_dim,))

        p.wy = np.random.normal(size=(state_dim, out_dim))
        p.by = np.zeros((out_dim,))

        return p

class Genome:
    def __init__(self) -> None:
        self.chromosomes = None
        self.info = None
        self.mutation_rate = None

    @classmethod
    def new(cls):

        mutation_rate = np.random.normal()
        

        size_factor = np.random.normal()
        sight_range = np.random.normal(0, 1)
        color = np.random.normal(0, 1, size=(3,))
        
        int_dim = 2 + 1 # pos, hp
        ext_dim = 2 + 1 # dist, type
        ext_att_n = 10
        state_dim = 32
        out_dim = 2# dir

        kernel_std = np.random.uniform(0, 0.1)
        bias_std = np.random.uniform(0, 0.01)

        kernel_init = lambda s : np.float16(np.random.normal(size=s) * kernel_std)
        #bias_init = lambda s : np.float16(np.zeros(s))
        bias_init = lambda s : np.float16(np.random.normal(size=s) * bias_std)

        w_int = kernel_init((int_dim, state_dim))
        b_int = bias_init((state_dim,))

        w_ext = kernel_init((ext_dim, state_dim))
        b_ext = bias_init((state_dim,))

        wc = kernel_init((state_dim*4, state_dim))
        bc = bias_init((state_dim,))
        
        wi = kernel_init((state_dim*4, state_dim))
        bi = bias_init(state_dim,)

        wf = kernel_init((state_dim*4, state_dim))
        bf = bias_init((state_dim,))

        wo = kernel_init((state_dim, state_dim))
        bo = bias_init((state_dim,))

        wy = kernel_init((state_dim, out_dim))
        by = bias_init((out_dim,))

        g = Genome()
        g.info = [int_dim, ext_dim, ext_att_n, state_dim, out_dim]
        g.chromosomes = [[size_factor], [sight_range], [color], [w_int, b_int, w_ext, b_ext, wc, bc, wi, bi, wf, bf, wo, bo, wy, by]]
        g.mutation_rate = mutation_rate
        return g
    
    @classmethod
    def from_parent(cls, parent):
        info = parent.info
        chromosomes = parent.chromosomes
        _mutation_rate = parent.mutation_rate
        mutation_rate = sigmoid(_mutation_rate)
        mutation_rate = mutation_rate * (2e-1 - 1e-5) + 1e-5

        def mutate(x):
            if type(x) is float:
                #return float(x * np.float16(np.random.normal(1)))
                return float(x + mutation_rate * np.float16(np.random.normal()))
            else:
                #return np.float16(x * np.random.normal(1, size=x.shape))
                return np.float16(x + mutation_rate * np.random.normal(size=x.shape))

        chromosomes = tree_util.tree_map(mutate, chromosomes)

        mutation_rate = float(_mutation_rate + _mutation_rate * 0.1 * np.random.normal())
        
        child = cls()
        child.info = info
        child.chromosomes = chromosomes
        child.mutation_rate = mutation_rate

        return child

class Body_with_owner(pymunk.body.Body):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner = None

class Creature:
    def __init__(self):
        self.batch = None
        self.space = None
        self.body_shape = None
        self.sight_shape = None
        self.shapes = None
        self.body = None
        self.genome = None
        
        self.draw_body_shape = None
        self.draw_sight_shape = None

        self.max_hit_points = None
        self.hit_points = None

        self.creatures_in_sight = []

    def _size_factor_from_genome(self):
        size_factor = self.genome.chromosomes[0][0]
        size_factor = sigmoid(size_factor)
        return size_factor
    
    def _sight_range_from_genome(self):
        sight_range = self.genome.chromosomes[1][0]
        sight_range = sigmoid(sight_range)
        sight_range = sight_range * (MAX_SIGHT - MIN_SIGHT) + MIN_SIGHT
        return sight_range
    
    def _color_from_genome(self):
        #color = self.genome.chromosomes[2][0]
        mr = self.genome.mutation_rate
        mr = sigmoid(mr)
        r = mr * 255
        g = 0
        b = (1-mr) * 255
        color = (r, g, b)
        
        color = np.rint(color)
        color = np.clip(color, 0, 255)
        color = [int(c) for c in color]
        return color
    
    def _brain_from_genome(self):
        params = self.genome.chromosomes[3]
        info = self.genome.info
        #self.state = np.float16(np.zeros((info[3])))
        self.state = [np.float16(np.zeros((info[3]))), np.float16(np.zeros((info[3])))]

        def fwd(params, xint, xext, state):
            w_int, b_int, w_ext, b_ext, wc, bc, wi, bi, wf, bf, wo, bo, wy, by = params

            #state = layernorm(state)
            ctm1, htm1 = state
            external = np.zeros((b_ext.shape))

            
            if len(xext) > 0:
                    
                for xe in xext:
                    external += xe @ w_ext + b_ext

                #external = external / len(xext)

            #external = layernorm(external)

            internal = xint @ w_int + b_int
            #internal = layernorm(internal)
            
            x = np.concatenate([internal, external, ctm1, htm1], -1)


            c = x @ wc + bc
            c = np.tanh(c)
            #c1, c2, c3 = np.split(c, 3)
            #c1 = layernorm(c1)
            #c2 = layernorm(c2)
            #c3 = layernorm(c3)
            #c = c1# * c2 + c3

            i = x @ wi + bi
            i = sigmoid(i)
            
            f = x @ wf + bf
            f = sigmoid(f)

            o = c @ wo + bo
            o = np.tanh(o)

            c = c * i + ctm1 * f

            h = o * c
            h = layernorm(h)
            
            y = h @ wy + by
            y = np.tanh(y)
            return y, [c, h]
        
        self._think = fwd

    def take_damage(self, value):
        self.hit_points -= value

    def maybe_eat_food(self, value):
        if self.hit_points < self.max_hit_points:

            self.hit_points += value
            self.hit_points = min(self.hit_points, self.max_hit_points)
            return True
        else:
            return False
        
    def _sense_internal(self):
        x,y = self.body.position

        x = x / window.width
        x = x * 2 - 1

        y = y / window.height
        y = y * 2 - 1

        hp = self.hit_points / self.max_hit_points

        internal = np.float16(np.array([x, y, hp]))

        return internal

    def _sense_external(self):

        ranked = sorted(self.creatures_in_sight, reverse=True, key = lambda tup : tup[1])[:self.genome.info[2]]
        ranked = [np.array([type_and_dist[0], type_and_dist[1][0], type_and_dist[1][1]]) for type_and_dist in ranked]
        self.creatures_in_sight.clear()

        return ranked

    def think(self):

        internal = self._sense_internal()
        external = self._sense_external()

        direction, state = self._think(self.genome.chromosomes[3], internal, external, self.state)

        self.state = state

        return direction

    def move(self, direction):
        force = self.body.mass * direction
        self.body.apply_impulse_at_local_point((float(force[0]), float(force[1])))

    @classmethod
    def new(cls, space, batch, genome, x, y):
        cr = Creature()
        cr.genome = genome
        cr.space = space
        cr.batch = batch

        brain = cr._brain_from_genome()
        
        size_factor = cr._size_factor_from_genome()
        sight_range = cr._sight_range_from_genome()
        color = cr._color_from_genome()
        radius = RADIUS_FROM_SIZE_FACTOR(size_factor)
        mass = MASS_FROM_SIZE_FACTOR(size_factor)
        hit_points = HP_FROM_SIZE_FACTOR(size_factor)

        cr.size_factor = size_factor

        cr.body = Body_with_owner(0,0)#, pymunk.body.Body.KINEMATIC)
        cr.body.owner = cr
        cr.body_shape = pymunk.shapes.Circle(cr.body, radius=radius)
        cr.body_shape.mass = mass
        cr.body_shape.density = 1
        cr.body_shape.collision_type = collision_types["creatures"]

        cr.sight_shape = pymunk.shapes.Circle(cr.body, radius=sight_range)
        cr.sight_shape.sensor = True
        cr.sight_shape.collision_type = collision_types["sight"]

        cr.max_hit_points = hit_points
        cr.hit_points = hit_points

        cr.shapes = [cr.body_shape, cr.sight_shape]

        cr.space.add(cr.body, *cr.shapes)
        cr.body.position = (x,y)
        cr.space.reindex_shapes_for_body(cr.body)

        cr.draw_body_shape = pyglet.shapes.Circle(cr.body.position.x, cr.body.position.y, 
                                cr.body_shape.radius, color=color, batch=cr.batch)
        
        cr.draw_sight_shape = pyglet.shapes.Circle(cr.body.position.x, cr.body.position.y,
                                cr.sight_shape.radius, color=(0, 255, 0, 33), batch=debug_batch)
        

        #cr._debug_draw_fn = lambda _cls : f"""{_cls.hit_points}, {round(_cls.body_shape.mass,2)}, {round(_cls.body.velocity.x, 2)}, {round(_cls.body.velocity.y, 2)}"""
        cr._debug_draw_fn = lambda _cls : f"""{_cls.hit_points}"""
        cr._debug_draw = pyglet.text.Label("", x=cr.body.position.x, y=cr.body.position.y, batch=debug_batch)
        

        return cr
    
    def update(self, dt):
        
        direction = self.think()
        self.move(direction)
        
        x,y = self.body.position
        
        self.draw_body_shape.x = x
        self.draw_body_shape.y = y

        self.draw_sight_shape.x = x
        self.draw_sight_shape.y = y    

        self._debug_draw.x = x
        self._debug_draw.y = y
        self._debug_draw.text = self._debug_draw_fn(self)

        self.creatures_in_sight = []

class Food:
    def __init__(self) -> None:
        self.value = None
        self.space = None
        self.batch = None

    @classmethod
    def new(cls, space, batch, value, x, y):
        food = cls()
        
        food.space = space
        food.batch = batch

        food.value = value
        food.body = Body_with_owner(0, 0, pymunk.body.Body.KINEMATIC)

        food.body.owner = food
        food.body_shape = pymunk.shapes.Circle(food.body, radius=40)
        food.body_shape.collision_type = collision_types["food"]

        food.space.add(food.body, food.body_shape)
        food.body.position = (x,y)
        food.space.reindex_shapes_for_body(food.body)

        food.draw_body_shape = pyglet.shapes.Circle(food.body.position.x, food.body.position.y, 
                                food.body_shape.radius, color=(0, 255, 0), batch=food.batch)
        
        food._debug_draw_fn = lambda _cls : f"""{_cls.value}"""
        food._debug_draw = pyglet.text.Label("", x=food.body.position.x, y=food.body.position.y, batch=debug_batch)

        return food

    def update(self, dt):
        self._debug_draw.text = self._debug_draw_fn(self)
        if self.value <= 0:
            pass

def init_space():
    
    space = pymunk.Space(threaded=True)
    space.threads = THREADS
    space.damping = 0.8




    margin = 0
    static_lines = [
        pymunk.Segment(space.static_body, (0+margin, 0+margin), (0+margin, window.height-margin), 2),
        pymunk.Segment(space.static_body, (0+margin, 0+margin), (window.width-margin, 0+margin), 2),
        pymunk.Segment(space.static_body, (window.width-margin, 0+margin), (window.width-margin, window.height-margin), 2),
        pymunk.Segment(space.static_body, (0+margin, window.height-margin), (window.width-margin, window.height-margin), 2),
    ]
    for line in static_lines:
        line.color = (100, 0, 0)
        line.elasticity = 1.0
        line.filter = pymunk.ShapeFilter(categories=1)

    space.add(*static_lines)
    return space


if __name__ == "__main__":


    space = init_space()

    n_iter = 0

    info_display = pyglet.text.Label("", x=window.width//100 * 1, y=window.height//100 * 95, color=(100, 100, 100, 255), batch=batch)

    add_creature_creature_collision(space)
    add_sight_creature_collision(space)
    add_creature_boundaries_collision(space)
    add_creature_food_collision(space)
    add_sight_food_collision(space)

    POP_SIZE = 400
    N_ELITE = 50

    N_FOODS = 10
    FOOD_VAL = 33

    STEPS = 0



    pop = {i : Creature.new(space, batch, Genome.new(), window.rnd_x(), window.rnd_y()) for i in range(POP_SIZE)}
    
    foods = {i : Food.new(space, batch, FOOD_VAL, window.rnd_x(), window.rnd_y()) for i in range(N_FOODS)}

    
    def update(dt):
        global pop
        global STEPS
        global foods
        global N_ELITE
        #global n_iter



        # maybe init new #
        if len(pop) <= N_ELITE:
            global n_iter
            n_iter += 1
            elite_genomes = [c.genome for c in pop.values()]

            for cr in pop.values():
                space.remove(cr.body, *cr.shapes)
                cr.draw_body_shape.delete()
                cr.draw_sight_shape.delete()
                cr._debug_draw.delete()

            pop = {}

            pc = 0
            
            for idx in range(POP_SIZE):
                parent = elite_genomes[pc]
                pc += 1
                if pc >= len(elite_genomes):
                    pc = 0

                child_genome = Genome.from_parent(parent)
                child = Creature.new(space, batch, child_genome, window.rnd_x(), window.rnd_y())
                pop[idx] = child


            for idx in foods:
                fd = foods[idx]
                fd.value = -1


        # remove dead 
        _pop = {}

        for idx in pop:
            cr = pop[idx]
            if cr.hit_points <= 0 and len(pop) > N_ELITE or window.pos_in_bounds(*cr.body.position) is False:
                space.remove(cr.body, *cr.shapes)
                cr.draw_body_shape.delete()
                cr.draw_sight_shape.delete()
                cr._debug_draw.delete()
            else:
                _pop[idx] = cr

        pop = _pop    

        _foods = {}

        for idx in foods:
            fd = foods[idx]
            if fd.value <= 0:
                space.remove(fd.body, fd.body_shape)
                fd.draw_body_shape.delete()
                fd._debug_draw.delete()
            else:
                _foods[idx] = fd

        foods = _foods

        info_display.text = f"{STEPS}, n_bodies: {len(pop)}, iteration: {n_iter}"
        
        for idx in pop:
            cr = pop[idx]
            cr.update(dt)

        space.step(dt)

        for idx in foods:
            fd = foods[idx]
            fd.update(dt)

        


        STEPS += 1
        if (STEPS+1) % 60 == 0:
            for idx in pop:
                cr = pop[idx]
                cr.take_damage(1)

        """if (STEPS+1) % 60 == 0:
            for idx in foods:
                fd = foods[idx]
                space.remove(fd.body, fd.body_shape)
                fd.draw_body_shape.delete()
                fd._debug_draw.delete()"""
        
        if len(foods) == 0:

            foods = {i : Food.new(space, batch, FOOD_VAL, window.rnd_x(), window.rnd_y()) for i in range(N_FOODS)}


            

    pyglet.clock.schedule_interval(update, 1/60.0) 
    event_loop = pyglet.app.EventLoop()
    pyglet.app.event_loop.run()
    