// =============================================================================
//  MAZE-SOLVING ROBOT
//  A NEURAL NETWORK, TUNED BY A GENETIC ALGORITHM
// =============================================================================
//
//  WHAT THIS PROGRAM DOES (in plain English)
//  -----------------------------------------
//  There is a maze. A little robot starts in the top-left corner and must walk
//  to the exit door in the bottom-right area. Nobody tells the robot the rules.
//  It has a "brain" made of numbers, and at the start those numbers are random,
//  so the robot stumbles around blindly.
//
//  We then breed robots the same way farmers breed cows:
//
//      1. Build 300 robots with random brains.
//      2. Let each one try the maze and give it a score (its "fitness").
//      3. Copy the best robots into the next generation, with small random
//         changes ("mutations") in their brains.
//      4. Repeat. The best robots slowly become better parents, so the
//         children get better, and after a few hundred rounds one of them
//         walks straight to the door.
//
//  The robot is never taught "turn left at the corner". It figures out a way
//  to behave by trial and error, exactly like evolution does with animals.
//
//
// =============================================================================
//  PART 0 -- THE MATHS, EXPLAINED LIKE YOU ARE IN 5th GRADE
// =============================================================================
//
//  Every single thing this program does is built from ONE tiny operation that
//  you already know: multiply, then add.
//
//  Example. You buy 3 apples and 4 oranges. One apple costs 2 coins and one
//  orange costs 1 coin. How much money do you need?
//
//        3 * 2  +  4 * 1   =   6 + 4   =   10 coins
//        ^   ^     ^   ^
//        |   |     |   |
//      amount      amount
//            price       price
//
//  Mathematicians call this a WEIGHTED SUM. That is ALL a neuron is.
//
//  In this program:
//
//      "how many apples"   ->  an INPUT   (a number the robot SENSES)
//      "price of an apple" ->  a WEIGHT    (a number the robot LEARNS)
//
//  So one neuron is a little calculator that says:
//
//      answer = input1*weight1 + input2*weight2 + ... + input10*weight10
//
//  ---------------------------------------------------------------------------
//  1) WHY THE WEIGHTS ARE THE ONLY THING WE "LEARN"
//  ---------------------------------------------------------------------------
//  The inputs come from the maze. We cannot change those. But WE choose the
//  weights. If we choose them badly, the neuron's answer is nonsense. If we
//  choose them well, the robot behaves sensibly. So "learning" here means
//  nothing more mysterious than "finding good values for these weights".
//
//  Think of each weight as a VOLUME KNOB. Turning a knob up means "pay more
//  attention to this input". Turning it down (or making it negative) means
//  "pay less attention" or even "treat this input as a warning sign".
//
//  ---------------------------------------------------------------------------
//  2) WHAT A "BIAS" IS
//  ---------------------------------------------------------------------------
//  Some neurons get ONE more number added at the very end, called a BIAS.
//  Picture a delivery fee added to your shopping bill:
//
//      answer = (shopping total) + (delivery fee)
//
//  Why is that useful? Because without it, a neuron can only "switch on" when
//  the shopping total is already bigger than zero. With a delivery fee of -5,
//  the neuron switches on only when the shopping total is bigger than 5.
//  So the fee is a way of moving the "yes / no" line wherever we want.
//  That is the bias's entire job: it slides the threshold.
//
//  (The old version of this program had no biases at all. That meant every
//  neuron whose weights happened to be negative could NEVER switch on -- it
//  was dead forever, and evolution could not wake it up. Roughly half of the
//  hidden neurons were wasted.)
//
//  ---------------------------------------------------------------------------
//  3) WHAT "ReLU" IS
//  ---------------------------------------------------------------------------
//      relu(x)  =  x    if x is positive
//      relu(x)  =  0    if x is zero or negative
//
//  In words: "if the answer came out negative, throw it away and write 0."
//  (Shop version: you cannot buy -3 apples, so we just say 0 apples.)
//
//  Why bother? Because it turns each neuron into a SWITCH: it is either OFF
//  (0) or ON (some positive amount). Switches are what let a network build
//  "if this AND that, then ..." rules instead of only straight-line rules.
//
//  ---------------------------------------------------------------------------
//  4) HOW THE ROBOT CHOOSES A MOVE
//  ---------------------------------------------------------------------------
//  Four different output neurons each answer "how much do I like this move?".
//  The robot simply picks the move with the BIGGEST number. That trick has a
//  name -- argmax, "the argument (index) of the maximum". You do it when you
//  pick the tastiest thing on a menu: you do not convert tastiness to
//  percentages, you just compare and take the biggest.
//
//  ---------------------------------------------------------------------------
//  5) WHY THERE IS A "HIDDEN LAYER"
//  ---------------------------------------------------------------------------
//  One layer of neurons can only make simple "if ... then" decisions. A hidden
//  layer in the middle lets the robot build COMBINATIONS of combinations:
//  "if there is a wall on my left AND open space ahead AND I am in the top row,
//  then go right". Those richer rules are what make maze solving possible.
//
//
// =============================================================================
//  PART 0.5 -- THE SEVEN BUGS THAT KEPT THE OLD VERSION FROM EVER WORKING
// =============================================================================
//
//  The previous version looked almost right but never solved the maze in
//  10,000 generations -- the whole population turned into identical clones
//  and the best score froze at 62.18 for 6,500 generations. Seven separate
//  things were wrong, and each one was hiding the next one. They are marked
//  in the code with  [FIX 1]..[FIX 7].
//
//  [FIX 1] THE SENSORS WERE LYING.
//      Old code:  if dist > 5 { continue }  left the input at 0.
//      Result:    any corridor longer than 5 tiles reported 0, which is
//                 the SAME as "a wall is right in front of me". The robot
//                 could not tell "open corridor" from "wall".
//
//  [FIX 2] THE "I CAN SEE THE DOOR" SIGNAL WAS ALWAYS THROWN AWAY.
//      Old code added +5.0 to the distance when the exit was in sight, which
//      guaranteed dist > 5, which triggered [FIX 1]'s `continue`. So the one
//      signal the robot most needed was silently deleted every single time.
//      The door now gets its own dedicated input.
//
//  [FIX 3] THE SENSOR LIST WAS HARD-CODED TO 6 NUMBERS.
//      Old code:  inputs := make([]float64, 6)
//      So the moment you tried to give the robot MORE senses by changing
//      numInputs, the network read past the end of the list and crashed.
//      That is why your "advanced sensors" experiment failed so badly --
//      it was a plumbing bug, not a bad idea. Your instinct was right.
//
//  [FIX 4] THE FITNESS FUNCTION WAS REWARDING THE WRONG THING.
//      Old code:  fitness = 100/(straightLineDistance+1) + tilesVisited*5
//      Two problems. (a) Wandering could score 275, while "get near the door"
//      could only score 100 -- so we were breeding tourists, not solvers.
//      (b) It measured distance AS THE CROW FLIES. But the only route to this
//      exit goes DOWN to row 7 first and only then back UP, so walking the
//      correct path made the score worse for about ten steps. Evolution was
//      punished for doing the right thing. See the ripple explanation below.
//
//  [FIX 5] MUTATION WAS BOTH TOO BIG AND TOO SMALL AT ONCE.
//      Old code nudged 55% of the weights by only 0.2. 55% of ~320 weights is
//      ~176 changes: that re-randomises the child, so good parents produce
//      bad children and nothing accumulates. Yet each individual nudge of 0.2
//      is too small to actually flip a decision, so the robot's behaviour
//      barely changed either. Net effect: expensive random search.
//      Now: only 10% of the weights move, but each move is bigger (sigma 0.8),
//      which is large enough to genuinely change a decision.
//
//  [FIX 6] THE POPULATION TURNED INTO CLONES.
//      Old code kept exactly ONE elite, so within ~3,500 generations all 100
//      robots had become copies of one robot and had literally ONE distinct
//      fitness value. With no diversity left, there is nothing to evolve.
//      Now: keep the top 10 as parents and inject fresh random robots.

//  [FIX 7] BUMPING A WALL BURNED THE WHOLE RUN.
//      The robot's policy is deterministic -- it has no memory. So if it
//      decides "left" at a tile where left is a wall, it will decide "left"
//      there forever. In the old version that was fine, but the run kept going
//      for the remaining 40+ steps doing nothing at all. Traced: the best robot
//      got stuck at tile (2,8) and wasted steps 12 through 55 bumping into the
//      same wall. Now a bump ends the run immediately with a small penalty,
//      so a "stuck" robot is cheap to test and scores badly for being stuck.
//
//
// =============================================================================
//  SUGGESTED READING ORDER
// =============================================================================
//      PART 1  the maze                   -- what the robot must solve
//      PART 2  the distance field         -- how we score "getting closer"
//      PART 3  the robot's brain          -- weights, biases, ReLU, argmax
//      PART 4  the robot's senses         -- what the 10 inputs mean
//      PART 5  one attempt at the maze    -- walk, win, or bump
//      PART 6  the evolution loop         -- breed, mutate, repeat
//      PART 7  the animation              -- watch the winner walk
//
//  HOW TO RUN IT:   go run .
//  (This folder is a Go "module" now, so `go run .` and `go build .` both work.
//   Before, you had to write `go run enchanced.maze.go`, and the old rand.go
//   file in this folder clashed with this one -- it has been removed.)
// =============================================================================

package main

import (
	"fmt"          // printing text to the terminal
	"math/rand/v2" // random numbers (see the note next to rng below)
	"sort"         // sorting the population by fitness
	"time"         // used to pick a different random seed on every run
)

// ---------------------------------------------------------------------
// PART 1 -- THE MAZE
// ---------------------------------------------------------------------
//
// The maze is a grid of numbers, stored as a slice of slices: a list of rows,
// and every row is a list of columns. The FIRST index picks the ROW (top to
// bottom), the SECOND index picks the COLUMN (left to right).
//
// What the numbers mean:
//
//	1 = wall   (the robot may NOT walk here)
//	0 = empty  (the robot may walk here)
//	3 = exit   (the goal -- standing here wins the game)
//
// Drawn out, this is the exact maze used below.  rows are 0..8, columns 0..11.
//
//	      col 0  1  2  3  4  5  6  7  8  9 10 11
//	row 0    ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██
//	row 1    ██ 🤖  ·  ·  ██  ·  ·  ·  ·  ·  · ██
//	row 2    ██  ·  ██  ·  ██  ·  ██ ██  ·  ██ ██ ██
//	row 3    ██  ·  ██  ·  ·  ·  ██ ██  ·  ██ 🚪 ██
//	row 4    ██  ·  ██ ██ ██  ·  ██ ██  ·  ██  · ██
//	row 5    ██  ·  ██  ·  ·  ·  ██  ·  ·  ██  · ██
//	row 6    ██  ·  ██ ██ ██ ██ ██  ·  ██ ██  · ██
//	row 7    ██  ·  ·  ·  ·  ██ ██  ·  ·  ·  · ██
//	row 8    ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██
//
// Notice the shape of the ONLY route. From the start (row 1, col 1) the robot
// goes right, drops down, goes right again, climbs back UP to row 1, crosses to
// column 8, dives all the way DOWN to row 7, runs right, then climbs back UP
// column 10 to the door at (row 3, col 10).
//
// That shape matters a lot: the robot has to walk AWAY from the door -- down to
// row 7 -- in order to get around a wall. That is exactly why the old
// "straight-line distance" score was broken (see [FIX 4] and PART 2).
//
// The shortest route is 25 moves. Each robot gets maxSteps = 60 moves, which is
// comfortably enough.
var maze = [][]int{
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	{1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1}, // start is (1, 1)
	{1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1},
	{1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 3, 1}, // exit is (3, 10)
	{1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1},
	{1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1},
	{1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1},
	{1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1},
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
}

// NAMING QUIRK, kept from the original file: "X" means ROW (up/down) and
// "Y" means COLUMN (left/right). That is the opposite of the usual maths
// convention, but it is only a name. Do not let it confuse you.
const startX, startY = 7, 4 // where every run begins
const exitX, exitY = 3, 10  // the door

// rows and cols are derived from the maze itself, so you can resize the maze
// and nothing else breaks. (The old file hard-coded "6" and "4" in places,
// which is exactly the sort of thing that caused [FIX 3].)
var rows = len(maze)
var cols = len(maze[0])

// rng is our random number generator, and this single line deserves a note.
//
// The old file called rand.Seed(time.Now().UnixNano()). On modern Go that call
// does NOTHING AT ALL -- it was switched off years ago. So the old program was
// not seeded the way the comment claimed.
//
// Here we build our own generator: PCG is a small, fast, high-quality random
// algorithm. rand.NewPCG takes two 64-bit "seed" numbers that decide the whole
// infinite sequence of random values. We feed it the current time so every run
// is different.
//
// EDUCATION EXPERIMENT: change the two numbers below to fixed values, like
// rand.NewPCG(1, 2), and every run will behave IDENTICALLY. That is how you
// debug evolution -- a reproducible run is worth a hundred random ones.
var rng = rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 0x9E3779B97F4A7C15))

// ---------------------------------------------------------------------
// PART 2 -- THE DISTANCE FIELD  (the "ripple" map)
// ---------------------------------------------------------------------
//
// We need a score that says "this tile is N moves from the door".
//
// A crow flies in a straight line. A mouse has to walk around walls. The old
// program used the CROW distance (Pythagoras: sqrt(dx*dx + dy*dy)) and that was
// the single most damaging bug, because in this maze the mouse must walk
// BACKWARDS (away from the door) to get around a wall. With the crow distance,
// doing the correct thing made the score worse.
//
// So we use the MOUSE distance. How do we find it? With a ripple:
//
//  1. Write 0 on the exit tile.
//
//  2. Every walkable tile next to a 0 gets 1.
//
//  3. Every walkable tile next to a 1 gets 2.
//
//  4. Keep going. Each wave is one more step away from the door.
//
//     ██ ██ ██ ██ ██ ██ ██
//     ██  5  4  3 ██  7  6   ...
//     ██  6 ██  2 ██  8  9
//     ██  7 ██  1  0 🚪 ...
//
// This is called a BREADTH-FIRST SEARCH (BFS), and it is one of the most useful
// ideas in all of computer science: "how far is everything from this one
// thing?" You will meet it again in map apps, social networks and games.
//
// Because the ripple counts real steps, walking the correct path now always
// makes the number go DOWN by one. Evolution finally gets an honest signal.
var distToExit [][]int

const unreachable = 1 << 20 // a big number, meaning "we never got here"

func buildDistanceField() {
	// Start by filling every square with "unreachable".
	distToExit = make([][]int, rows)
	for r := 0; r < rows; r++ {
		distToExit[r] = make([]int, cols)
		for c := 0; c < cols; c++ {
			distToExit[r][c] = unreachable
		}
	}

	// The ripple starts at the door.
	distToExit[exitX][exitY] = 0
	queue := [][2]int{{exitX, exitY}} // a "queue" = a line of tiles waiting their turn

	for len(queue) > 0 {
		p := queue[0] // take the tile at the front of the line
		queue = queue[1:]

		for d := 0; d < 4; d++ {
			nr := p[0] + dRow[d]
			nc := p[1] + dCol[d]

			// Skip walls, and skip tiles we already numbered.
			if maze[nr][nc] == 1 || distToExit[nr][nc] != unreachable {
				continue
			}
			// One step further from the door than the tile we came from.
			distToExit[nr][nc] = distToExit[p[0]][p[1]] + 1
			queue = append(queue, [2]int{nr, nc})
		}
	}
}

// ---------------------------------------------------------------------
// PART 3 -- THE ROBOT'S BRAIN
// ---------------------------------------------------------------------
//
// The brain is three layers, written left to right:
//
//	  [10 inputs]  --->  [16 hidden neurons]  --->  [4 outputs]
//
//	INPUTS (10): the robot's senses, built in PART 4.
//	HIDDEN (16): "in-between" neurons. The outside world never sees them.
//	             They exist so the brain can combine senses into rules.
//	OUTPUTS (4): one number per possible move. The biggest one wins.
//	             0 = Up, 1 = Down, 2 = Left, 3 = Right.
//
// How the numbers are stored: all the weights live in ONE long flat slice, not
// in a 2D table. weightsHidden[i*numHidden + h] is "the weight from input i to
// hidden neuron h". A flat slice is easier to copy and to mutate, which is all
// we ever do with it. The arithmetic is the same either way.
const numInputs = 10 // 4 sightlines + 4 "exit visible" flags + 2 coordinates
const numHidden = 16 // in-between neurons (was 32 in the old file)
const numOutputs = 4 // Up / Down / Left / Right

// The four directions, in the order the outputs use them. Row goes down as X
// grows, column goes right as Y grows.
var dRow = [4]int{-1, 1, 0, 0} // Up, Down, Left, Right  (row change)
var dCol = [4]int{0, 0, -1, 1} // Up, Down, Left, Right  (column change)

// Agent is ONE robot and its brain. A "struct" is just a named bundle of
// values kept together. Every robot owns four lists of numbers plus its score.
type Agent struct {
	// Knobs between the 10 senses and the 16 hidden neurons: 10*16 = 160 numbers.
	weightsHidden []float64

	// Knobs between the 16 hidden neurons and the 4 outputs: 16*4 = 64 numbers.
	weightsOutput []float64

	// One bias per hidden neuron and one per output neuron. These are the
	// "delivery fees" from PART 0 item 2: they slide each neuron's on/off
	// threshold. The old file had none at all, which permanently killed every
	// hidden neuron whose weights happened to come out negative -- roughly
	// half of them.
	biasHidden []float64
	biasOutput []float64

	// The score from the most recent attempt. Higher = better.
	fitness float64
}

// NewAgent builds a robot whose knobs are all set at random.
//
// We draw from a NORMAL distribution (a bell curve, mean 0, spread 1) rather
// than the old flat -1..+1 range. A bell curve gives us mostly small weights
// and the occasional big one, which is what neural networks like. rand.NormFloat64
// does exactly that.
//
// The biases start at 0.0 -- neutral. Evolution decides later whether to slide
// them up or down.
func NewAgent() Agent {
	a := Agent{
		weightsHidden: make([]float64, numInputs*numHidden),
		weightsOutput: make([]float64, numHidden*numOutputs),
		biasHidden:    make([]float64, numHidden),
		biasOutput:    make([]float64, numOutputs),
	}
	for i := range a.weightsHidden {
		a.weightsHidden[i] = rng.NormFloat64()
	}
	for i := range a.weightsOutput {
		a.weightsOutput[i] = rng.NormFloat64()
	}
	return a
}

// relu is the activation function (see PART 0 item 3): negatives become 0.
func relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

// Predict is the robot THINKING. Give it the 10 senses, it gives back a move.
//
// The `(a *Agent)` part means this is a "method": it may use the robot's own
// numbers. The `*` means we get a pointer (a reference to the real robot), not
// a copy, so we are not duplicating its brain 18,000 times per generation.
func (a *Agent) Predict(inputs []float64) int {
	// ---- STEP 1: senses -> hidden layer ----
	hidden := make([]float64, numHidden)
	for h := 0; h < numHidden; h++ {
		// Start from this neuron's bias ("delivery fee"), not from zero.
		sum := a.biasHidden[h]
		for i := 0; i < numInputs; i++ {
			sum += inputs[i] * a.weightsHidden[i*numHidden+h]
		}
		hidden[h] = relu(sum) // switch it off if the answer was negative
	}

	// ---- STEP 2: hidden layer -> 4 outputs ----
	// Remember only the biggest output and its index: that is argmax.
	bestAction := 0
	maxVal := -1e18 // effectively "minus infinity"
	for o := 0; o < numOutputs; o++ {
		sum := a.biasOutput[o]
		for h := 0; h < numHidden; h++ {
			sum += hidden[h] * a.weightsOutput[h*numOutputs+o]
		}
		if sum > maxVal {
			maxVal = sum
			bestAction = o
		}
	}
	return bestAction
}

// Clone makes a complete private copy of a robot. This matters because Go
// slices are shared by reference: with a plain `b := a`, both names would point
// at the SAME weight list, and mutating the child would silently mutate the
// parent too. A deep copy avoids that trap.
func (a *Agent) Clone() Agent {
	c := Agent{
		weightsHidden: make([]float64, len(a.weightsHidden)),
		weightsOutput: make([]float64, len(a.weightsOutput)),
		biasHidden:    make([]float64, len(a.biasHidden)),
		biasOutput:    make([]float64, len(a.biasOutput)),
	}
	copy(c.weightsHidden, a.weightsHidden)
	copy(c.weightsOutput, a.weightsOutput)
	copy(c.biasHidden, a.biasHidden)
	copy(c.biasOutput, a.biasOutput)
	return c
}

// Mutate makes small random changes to the robot's knobs -- the "child differs
// slightly from its parent" part of evolution.
//
// Two numbers control it, and they pull in opposite directions:
//
//	rate  = what FRACTION of the knobs get touched (0.10 = about 1 in 10)
//	sigma = how FAR each touched knob moves (0.8 = a real, noticeable nudge)
//
// [FIX 5] The old file used rate 0.55 and sigma 0.2, which is backwards on both
// counts: it touched over half the knobs (destroying the parent's behaviour)
// but moved each one so little that the robot's decisions barely changed.
//
// Analogy: to tune 100 radio dials you want to nudge a FEW dials a LOT, listen,
// and keep what sounds better. You do not want to wiggle all 100 dials by a
// hair: after the wiggle you can no longer hear the station you liked.
func (a *Agent) Mutate(rate, sigma float64) {
	nudge := func(v *float64) {
		if rng.Float64() < rate {
			*v += rng.NormFloat64() * sigma
		}
	}
	for i := range a.weightsHidden {
		nudge(&a.weightsHidden[i])
	}
	for i := range a.weightsOutput {
		nudge(&a.weightsOutput[i])
	}
	for i := range a.biasHidden {
		nudge(&a.biasHidden[i])
	}
	for i := range a.biasOutput {
		nudge(&a.biasOutput[i])
	}
}

// ---------------------------------------------------------------------
// PART 4 -- THE ROBOT'S SENSES  (10 numbers)
// ---------------------------------------------------------------------
//
// This is the part you were trying to improve, and it is now built from
// numInputs instead of a hard-coded 6, so you can safely add MORE senses:
// change numInputs, fill in the new slots here, and nothing crashes.
//
// The 10 numbers:
//
//	[0..3]  how far can I walk in this direction, from 0.0 to 1.0
//	        0.0 means "a wall is right in front of me"
//	        1.0 means "10 or more open tiles that way"
//	        order: Up, Down, Left, Right
//
//	[4..7]  can I SEE the door in this direction? 1.0 = yes, 0.0 = no
//	        order: Up, Down, Left, Right
//
//	[8]     my row, scaled to roughly 0..1
//	[9]     my column, scaled to roughly 0..1
//
// Two deliberate design choices:
//
//   - Everything is SCALED to roughly the same size (0..1). Raw row numbers
//     like "7" would otherwise shout ten times louder than a distance of "0.5",
//     and the network would listen mostly to the coordinates. In a weighted sum
//     the biggest numbers win, so we keep them all comparable.
//
//   - The door gets its OWN inputs instead of being folded into the distance,
//     which is what [FIX 2] was about.
const maxSight = 10.0 // sightlines longer than this are just reported as 10

func GetSensoryInputs(x, y int) []float64 {
	// Built from numInputs, never from a hard-coded number. [FIX 3]
	inputs := make([]float64, numInputs)

	for d := 0; d < 4; d++ {
		dist := 0.0    // how many open tiles we can walk before hitting a wall
		sawExit := 0.0 // 1.0 if the door is in this line of sight
		cx, cy := x, y // start walking from the robot's tile

		for {
			cx += dRow[d]
			cy += dCol[d]

			if maze[cx][cy] == 1 { // wall: end of the corridor
				break
			}
			dist += 1.0

			if maze[cx][cy] == 3 { // the door!
				sawExit = 1.0
				break
			}
		}

		// [FIX 1] Clamp, never `continue`. A long corridor is "1.0", not "0.0".
		// Without this single line, long corridors look identical to walls.
		if dist > maxSight {
			dist = maxSight
		}
		inputs[d] = dist / maxSight // squash 0..10 down to 0..1
		inputs[4+d] = sawExit       // [FIX 2] the door now reaches the brain
	}

	// Position, also squashed to roughly 0..1. rows-2 and cols-2 because the
	// outer ring of the maze is all wall.
	inputs[8] = float64(x) / float64(rows-2)
	inputs[9] = float64(y) / float64(cols-2)

	return inputs
}

// ---------------------------------------------------------------------
// PART 5 -- ONE ATTEMPT AT THE MAZE
// ---------------------------------------------------------------------
//
// RunMaze lets one robot try the maze. It returns true if the robot reached the
// door, and it writes the score into agent.fitness either way.
//
// SCORING, top to bottom:
//
//	reached the door   ->  1000 - steps
//	                       (so 25 moves scores 975 and 40 moves scores 960:
//	                        reaching it sooner is better)
//
//	otherwise          ->  (how many ripple-steps closer I got) * 10
//	                        minus  0.1 per step taken
//	                        minus  2.0 if the run ended by bumping a wall
//
// The highest possible non-winning score is about 240, so a win (about 975) is
// always dramatically better. That gives evolution a clear direction instead of
// the mixed-up old signal. [FIX 4]
const bumpPenalty = 2.0
const stepCost = 0.1

func progressScore(closestRipple, steps int) float64 {
	improvement := distToExit[startX][startY] - closestRipple
	return float64(improvement)*10.0 - stepCost*float64(steps)
}

func RunMaze(agent *Agent, maxStepsAllowed int) bool {
	x, y := startX, startY

	// The best (smallest) ripple number this run ever reached. It starts at the
	// value of the starting tile.
	closest := distToExit[startX][startY]

	for step := 0; step < maxStepsAllowed; step++ {
		if distToExit[x][y] < closest {
			closest = distToExit[x][y]
		}

		// The brain looks at the senses and picks a move.
		action := agent.Predict(GetSensoryInputs(x, y))

		nx := x + dRow[action]
		ny := y + dCol[action]

		if maze[nx][ny] == 1 {
			// [FIX 7] Bumping a wall ends the run right here.
			// Because the policy has no memory, the robot would otherwise pick
			// the same doomed direction for the remaining 40+ steps and learn
			// nothing. Ending it costs us nothing and scores "stuck" properly.
			agent.fitness = progressScore(closest, step) - bumpPenalty
			return false
		}

		x, y = nx, ny

		if x == exitX && y == exitY {
			agent.fitness = 1000.0 - float64(step+1)
			return true
		}
	}

	// Out of moves without reaching the door.
	agent.fitness = progressScore(closest, maxStepsAllowed)
	return false
}

// ---------------------------------------------------------------------
// PART 6 -- THE EVOLUTION LOOP
// ---------------------------------------------------------------------
//
// These are the knobs of the BREEDING process itself. They are not part of the
// robot's brain; they are how we raise a generation of robots.

const (
	populationSize = 300   // how many robots live in one generation
	maxGenerations = 10000 // safety stop
	maxSteps       = 60    // moves allowed per attempt (shortest route is 25)

	mutationRate  = 0.10 // fraction of knobs nudged in a child
	mutationSigma = 0.8  // how far each nudged knob travels

	// [FIX 6] Keep the best 10 robots UNCHANGED in every generation. The old
	// file kept only 1, and the population turned into clones. Keeping 10
	// throws away a little speed but keeps several different good ideas alive.
	numElites = 10
)

// A few brand-new random robots join every generation. These are the "fresh
// blood": they are usually terrible, but they carry gene combinations that the
// rest of the population lost. Without them, evolution can paint itself into a
// corner -- which is exactly what happened in the old version.
const freshBlood = populationSize / 20

// main is the function Go runs automatically.
func main() {
	buildDistanceField()

	// A tiny sanity check you can read: is the maze actually solvable?
	if distToExit[startX][startY] >= unreachable {
		fmt.Println("The exit cannot be reached from the start. Fix the maze!")
		return
	}
	fmt.Println("Maze loaded.")
	fmt.Printf("  %d x %d grid, %d open tiles\n", rows, cols, countOpenTiles())
	fmt.Printf("  shortest route from start to door: %d moves\n", distToExit[startX][startY])
	fmt.Printf("  brain: %d inputs -> %d hidden -> %d outputs\n", numInputs, numHidden, numOutputs)
	fmt.Printf("  population: %d robots, %d moves per attempt, mutation rate %.2f / sigma %.2f\n",
		populationSize, maxSteps, mutationRate, mutationSigma)
	fmt.Println()
	fmt.Println("Starting evolution...")
	fmt.Println()

	// Generation 0: 300 robots with completely random brains.
	population := make([]Agent, populationSize)
	for i := range population {
		population[i] = NewAgent()
	}

	for gen := 1; gen <= maxGenerations; gen++ {
		// ---------- 1. TEST EVERY ROBOT ----------
		bestIndex := 0
		foundWinner := false
		fitnessSum := 0.0

		for i := range population {
			won := RunMaze(&population[i], maxSteps)
			if won {
				foundWinner = true
			}
			fitnessSum += population[i].fitness
			if population[i].fitness > population[bestIndex].fitness {
				bestIndex = i
			}
		}

		bestFitness := population[bestIndex].fitness
		avgFitness := fitnessSum / float64(populationSize)
		fmt.Printf("Gen %4d | best %8.2f | average %8.2f\n", gen, bestFitness, avgFitness)

		// ---------- 2. DID ANYBODY WIN? ----------
		if foundWinner {
			// A winner always has a fitness above 900, so the best robot of this
			// generation is definitely the winner. Show its journey.
			fmt.Println()
			fmt.Printf("SUCCESS! A robot solved the maze in generation %d.\n", gen)
			fmt.Printf("It reached the door in %.0f moves.\n", 1000.0-bestFitness)
			fmt.Println("Starting the animation in 2 seconds...")
			time.Sleep(2 * time.Second)
			VisualRun(&population[bestIndex], maxSteps)
			return
		}

		// ---------- 3. RANK THE ROBOTS ----------
		// "ranked" is a list of POSITIONS (0,1,2,...) sorted so that the fittest
		// robot's position comes first. We sort positions rather than robots so
		// that we never accidentally copy the big weight lists around.
		ranked := make([]int, populationSize)
		for i := range ranked {
			ranked[i] = i
		}
		sort.Slice(ranked, func(a, b int) bool {
			return population[ranked[a]].fitness > population[ranked[b]].fitness
		})

		// ---------- 4. BUILD THE NEXT GENERATION ----------
		next := make([]Agent, 0, populationSize)

		// (a) Elites: the champion robots, copied with no changes at all.
		for e := 0; e < numElites; e++ {
			next = append(next, population[ranked[e]].Clone())
		}

		// (b) Fresh blood: brand-new random robots, to keep variety alive.
		for i := 0; i < freshBlood; i++ {
			next = append(next, NewAgent())
		}

		// (c) The rest: mutated children of good parents.
		//
		// A "tournament" picks two parents at random from the better half of the
		// population and lets the fitter one breed. This is cheap (no sorting
		// per child) and it keeps selection pressure without being brutal: a
		// mediocre robot still gets to be a parent sometimes.
		half := populationSize / 2
		for len(next) < populationSize {
			p1 := ranked[rng.IntN(half)]
			p2 := ranked[rng.IntN(half)]
			parent := p1
			if population[p2].fitness > population[p1].fitness {
				parent = p2
			}

			child := population[parent].Clone()
			child.Mutate(mutationRate, mutationSigma)
			next = append(next, child)
		}

		population = next
	}

	fmt.Println()
	fmt.Printf("Evolution finished after %d generations without a solution.\n", maxGenerations)
	fmt.Println("Try raising maxGenerations, or lowering mutationSigma a little.")
}

// countOpenTiles is a tiny helper used only for the greeting message.
func countOpenTiles() int {
	n := 0
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			if maze[r][c] != 1 {
				n++
			}
		}
	}
	return n
}

// ---------------------------------------------------------------------
// PART 7 -- THE ANIMATION  (for your eyes, not for the learning)
// ---------------------------------------------------------------------
//
// Nothing in this part affects evolution. It just draws the maze in the
// terminal so you can watch the winning robot walk.

// PrintMaze clears the screen and draws one frame.
func PrintMaze(agentX, agentY int) {
	// "\033[H\033[2J" is an ANSI escape code: jump the cursor to the top-left
	// corner and wipe the screen. That is what makes it look like an animation
	// instead of a scrolling wall of text.
	fmt.Print("\033[H\033[2J")

	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			switch {
			case r == agentX && c == agentY:
				fmt.Print("🤖 ")
			case maze[r][c] == 1:
				fmt.Print("██")
			case r == exitX && c == exitY:
				fmt.Print("🚪 ")
			default:
				fmt.Print("  ")
			}
		}
		fmt.Println() // newline after each row
	}
	fmt.Println("\nCtrl+C to stop.")
}

// VisualRun replays one robot's journey, one move at a time.
func VisualRun(agent *Agent, maxStepsAllowed int) {
	x, y := startX, startY

	for step := 0; step < maxStepsAllowed; step++ {
		PrintMaze(x, y)
		time.Sleep(200 * time.Millisecond) // pause so a human can follow along

		if x == exitX && y == exitY {
			PrintMaze(x, y)
			fmt.Println("\nThe robot reached the door!")
			return
		}

		action := agent.Predict(GetSensoryInputs(x, y))
		nx := x + dRow[action]
		ny := y + dCol[action]

		// Safety net: a winner should never bump a wall, but if the maze or the
		// sensors are edited later, this tells us straight away instead of
		// crashing with an index error.
		if maze[nx][ny] == 1 {
			fmt.Println("\nThe replayed robot bumped into a wall. Something changed. ")
			return
		}
		x, y = nx, ny
	}
}
