// =============================================================================
//  MAZE SOLVING ROBOT — A NEURAL NETWORK EVOLVED BY A GENETIC ALGORITHM
// =============================================================================
//
//  WHAT THIS PROGRAM DOES (in plain English):
//
//      We have a little maze: a grid of walls and open paths. We want to build
//      a "robot brain" that can walk from the starting tile to the exit door,
//      WITHOUT anyone telling it HOW. The robot only senses a few numbers about
//      its surroundings (how far the walls are) and must pick one of four moves:
//      Up, Down, Left, or Right.
//
//      The robot's "brain" is a NEURAL NETWORK. Think of it as a machine full
//      of tiny knobs (called "weights"). Each knob controls how strongly one
//      piece of information influences the robot's decision. At the start all
//      the knobs are set to random values, so the robot moves completely at
//      random.
//
//      We then create 100 such robots (a "population"), test each one in the
//      maze, and give each a "fitness" score for how well it did. The best
//      robots are copied into the next generation, and their knobs get small
//      random tweaks (this is "mutation" — the same idea as biological
//      evolution). Over many generations the random knobs slowly get tuned
//      until a robot can actually reach the exit.
//
//      This combination — a NEURAL NETWORK tuned by a GENETIC ALGORITHM — is
//      one of the simplest ways to make a computer "learn" without ever
//      teaching it any explicit rules.
//
//  WHAT YOU NEED TO KNOW ABOUT THE GO LANGUAGE TO READ THIS FILE:
//
//      - `func name(...) ... { ... }`  defines a function (a reusable block of
//        instructions). `main` is the function that runs automatically.
//      - `var x = 5`  declares a variable (a named box that stores a value).
//      - `const x = 5` declares a constant (a named box whose value never
//        changes while the program runs).
//      - `//` starts a comment; Go ignores the rest of that line.
//      - `[]int` is a "slice" — an ordered list of whole numbers like [1,2,3].
//      - `[][]int` is a list of lists — here it represents a 2D grid (rows).
//      - `for` is a loop — it repeats the instructions inside its braces { }.
//      - `if ... else` lets the program choose between two paths.
//      - `type Name struct { ... }` defines a custom object that groups several
//        values together (a box containing smaller labelled boxes).
//      - `map[string]int` is a dictionary: a set of key -> value pairs.
//
//  GLOSSARY OF THE "AI" WORDS USED IN THIS FILE:
//
//      NEURAL NETWORK : a stack of simple math operations (loosely inspired by
//                       brain cells) that maps input numbers to output numbers.
//      NEURON         : one unit in the network that adds up its inputs and
//                       produces a single output number.
//      WEIGHT         : a number (a "knob") that scales an input. Tuning the
//                       weights is exactly what "learning" means here.
//      INPUT          : the numbers the robot senses. There are 6 of them
//                       (4 wall-distances + its own row + its own column).
//      HIDDEN LAYER   : 16 extra neurons between inputs and outputs. They let
//                       the brain express more complex behaviour.
//      OUTPUT         : the robot's decision. There are 4 (Up/Down/Left/Right);
//                       the largest output wins.
//      ACTIVATION     : a function applied to a neuron's sum so the network can
//                       learn non-trivial patterns. We use one called ReLU.
//      FITNESS        : a score telling us how good a robot is. Higher = better.
//      GENERATION     : one round of testing all robots and breeding the best.
//      MUTATION       : a small random change to a weight, imitating evolution.
//      SELECTION      : picking the best robots to become parents.
//
//  SUGGESTED READING ORDER (if you get lost, follow this path):
//      1. The maze            (section 1)  — what the robot has to solve.
//      2. The neural network  (section 2)  — the robot's brain.
//      3. The simulation      (section 3)  — how the robot moves & gets scored.
//      4. The main loop       (section 4)  — how evolution runs the show.
//      5. Visualization       (section 5)  — how we watch the winner move.
//
// =============================================================================

package main

import (
	"fmt"       // "format" — used for printing text to the terminal.
	"math"      // math helpers: square root (Sqrt) and the largest possible float.
	"math/rand" // random numbers — used to scramble the initial weights.
	"time"      // clocks — used to seed randomness and to slow the animation down.
)

// ---------------------------------------------------------------------
// 1. MAZE DEFINITION
// ---------------------------------------------------------------------
//
// The maze is a grid of numbers stored in a "slice of slices": a list of rows,
// and each row is a list of columns. The FIRST index picks the ROW (top to
// bottom) and the SECOND index picks the COLUMN (left to right).
//
// The meaning of each number:
//
//	1 = Wall   (the robot may NOT walk here)
//	0 = Empty  (the robot may walk here)
//	3 = Exit   (the goal — reaching this tile wins the game)
//
// Drawn with its coordinates, the maze looks like this:
//
//	     col 0   1   2   3   4   5   6
//	row 0  ██  ██  ██  ██  ██  ██  ██
//	row 1  ██  🤖   .   .   ██   .   ██   <- Start is at row 1, col 1
//	row 2  ██   .   ██   .   ██   .   ██
//	row 3  ██   .   ██   .   .    .   ██
//	row 4  ██  ██  ██  ██  ██   .   ██
//	row 5  ██  ██  🚪   .   .    .   ██   <- Exit is at row 5, col 2
//	row 6  ██  ██  ██  ██  ██  ██  ██
//
// (The "3" sits at row 5, column 2 — which matches the exitX/exitY constants
// declared just below.)
var maze = [][]int{
	{1, 1, 1, 1, 1, 1, 1},
	{1, 0, 0, 0, 1, 0, 1}, // Start at (row 1, col 1) — value 0 = empty
	{1, 0, 1, 0, 1, 0, 1},
	{1, 0, 1, 0, 0, 0, 1},
	{1, 1, 1, 1, 1, 0, 1},
	{1, 1, 3, 0, 0, 0, 1}, // Exit at (row 5, col 2) — value 3 = exit
	{1, 1, 1, 1, 1, 1, 1},
}

// startX/startY is the robot's starting tile (row 1, column 1).
// exitX/exitY  is the goal tile (row 5, column 2).
//
// IMPORTANT NAMING QUIRK: in this program "X" means ROW (the vertical/top-to-
// bottom direction) and "Y" means COLUMN (the horizontal/left-to-right
// direction). The names are just labels the author chose; don't let the usual
// math meaning of "x is horizontal" confuse you here.
const startX, startY = 1, 1
const exitX, exitY = 5, 2

// ---------------------------------------------------------------------
// 2. NEURAL NETWORK STUFF
// ---------------------------------------------------------------------
//
// A neural network takes a list of input numbers and produces a list of output
// numbers. In between, it performs a fixed set of multiplications and
// additions. The numbers it multiplies by are called WEIGHTS — and those are
// the ONLY thing that gets "learned". All the rest is fixed code.
//
// Our network has THREE layers, written as input -> hidden -> output:
//
//     [6 inputs]  --->  [16 hidden neurons]  --->  [4 outputs]
//
//   * INPUTS (6): the robot's senses, built by the GetSensoryInputs function:
//        [0] distance to the wall going Up
//        [1] distance to the wall going Down
//        [2] distance to the wall going Left
//        [3] distance to the wall going Right
//        [4] the robot's current row    (X)
//        [5] the robot's current column (Y)
//
//   * HIDDEN (16): "in-between" neurons. The outside world never sees them;
//        they just let the brain combine the inputs in richer, more complex
//        ways. More hidden neurons = a more powerful (but slower to learn)
//        brain.
//
//   * OUTPUTS (4): one number per possible move. Whichever output is the
//        biggest becomes the chosen move: 0=Up, 1=Down, 2=Left, 3=Right.

const numInputs = 6  // how many input numbers the robot senses
const numHidden = 16 // how many neurons in the hidden layer
const numOutputs = 4 // how many moves to choose from (Up/Down/Left/Right)

// Agent represents ONE robot and its brain. A "struct" is just a named group
// of values bundled together. Each robot owns:
type Agent struct {
	// weightsHidden: the knobs between the 6 inputs and the 16 hidden neurons.
	// There are 6*16 = 96 of them, stored one after another in one flat list
	// (a "slice") instead of a 2D table. `float64` means "a decimal number".
	weightsHidden []float64

	// weightsOutput: the knobs between the 16 hidden neurons and the 4 outputs.
	// There are 16*4 = 64 of them.
	weightsOutput []float64

	// fitness: the robot's score from its most recent test run. Higher = better.
	fitness float64
}

// NewAgent creates a brand-new robot whose brain knobs (weights) are all set
// to random numbers between -1.0 and +1.0. Because the weights start random,
// this robot behaves completely randomly at first. Evolution is what later
// tunes these weights into something useful.
func NewAgent() Agent {
	// make() allocates a slice (a list) of the given length, filled with 0s.
	// Here we create one flat list for the hidden weights and one for the
	// output weights, ready to hold 96 and 64 numbers respectively.
	a := Agent{
		weightsHidden: make([]float64, numInputs*numHidden),
		weightsOutput: make([]float64, numHidden*numOutputs),
	}

	// "for i := range slice" loops once per element; i is the position (0,1,2,...).
	// rand.Float64() returns a random decimal in [0.0, 1.0). Multiplying by 2
	// gives [0.0, 2.0); subtracting 1 shifts that to [-1.0, 1.0). So each
	// weight lands somewhere between -1 and +1.
	for i := range a.weightsHidden {
		a.weightsHidden[i] = rand.Float64()*2 - 1
	}
	for i := range a.weightsOutput {
		a.weightsOutput[i] = rand.Float64()*2 - 1
	}
	return a
}

// relu is our ACTIVATION FUNCTION. A neuron first adds up all its weighted
// inputs into one number; relu then decides the neuron's final output:
//
//	relu(x) = x   if x is positive
//	relu(x) = 0   if x is negative
//
// In other words: any negative sum is clamped to zero, while positive sums
// pass through unchanged. "ReLU" stands for "Rectified Linear Unit". It is a
// very common, very cheap choice because it lets the network learn non-linear
// (i.e. not just straight-line) behaviour, which is required for interesting
// decision-making.
func relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

// Predict is the robot "thinking". Given the 6 sensory inputs, it runs them
// forward through the network (a "forward pass") and returns the number of the
// move it wants to make: 0=Up, 1=Down, 2=Left, 3=Right.
//
// The `(a *Agent)` part means this is a "method" — it can read and use the
// robot's own weights. The `*` means we receive a POINTER (a reference to the
// original robot), not a copy.
func (a *Agent) Predict(inputs []float64) int {
	// hidden will hold the 16 hidden neurons' values for this one step.
	hidden := make([]float64, numHidden)

	// --- STEP 1: INPUTS -> HIDDEN ---------------------------------------
	// For each hidden neuron h, compute a "weighted sum" of all 6 inputs:
	//     sum = input0*w0 + input1*w1 + ... + input5*w5
	// The weight layout: weightsHidden[i*numHidden + h] is the weight that
	// connects input #i to hidden neuron #h. (They are stored in one flat list,
	// grouped so that all weights feeding neuron h sit together.)
	for h := 0; h < numHidden; h++ {
		sum := 0.0
		for i := 0; i < numInputs; i++ {
			sum += inputs[i] * a.weightsHidden[i*numHidden+h]
		}
		hidden[h] = relu(sum) // apply the activation to the sum
	}

	// --- STEP 2: HIDDEN -> OUTPUT ---------------------------------------
	// Each of the 4 outputs is also a weighted sum of the hidden neurons.
	// We do NOT use relu here; instead we simply find the BIGGEST output,
	// because a bigger output means "I'm more confident about this move".
	bestAction := 0            // remembers which output index is largest
	maxVal := -math.MaxFloat64 // start from the smallest possible number
	for o := 0; o < numOutputs; o++ {
		sum := 0.0
		for h := 0; h < numHidden; h++ {
			// weight connecting hidden neuron #h to output #o
			sum += hidden[h] * a.weightsOutput[h*numOutputs+o]
		}
		// If this output beats the best we've seen so far, remember it.
		if sum > maxVal {
			maxVal = sum
			bestAction = o
		}
	}
	return bestAction
}

// Clone makes an exact copy of a robot. This matters because slices in Go are
// shared by REFERENCE: if we simply wrote `copy2 := copy1`, both names would
// point at the SAME weight lists, so changing one would silently change the
// other. A proper "deep copy" gives the new robot its own private weight lists.
func (a *Agent) Clone() Agent {
	// Create fresh, empty lists with the same lengths as the original's lists.
	c := Agent{
		weightsHidden: make([]float64, len(a.weightsHidden)),
		weightsOutput: make([]float64, len(a.weightsOutput)),
	}
	// copy() copies elements from the second slice into the first slice.
	copy(c.weightsHidden, a.weightsHidden)
	copy(c.weightsOutput, a.weightsOutput)
	return c
}

// Mutate makes small random changes ("mutations") to some of the robot's
// weights. For every single weight we roll a random number; if it is below
// `rate`, we nudge that weight by a random amount drawn from a normal
// distribution (a bell curve centred on 0) and scaled by 0.2.
//
// `rate` is therefore the PROBABILITY that any one weight gets changed. A small
// rate keeps most of the parent's behaviour intact while exploring tiny
// variations around it — exactly like children being slightly different from
// their parents in nature.
func (a *Agent) Mutate(rate float64) {
	// Walk through every hidden weight and maybe nudge it.
	for i := range a.weightsHidden {
		if rand.Float64() < rate {
			a.weightsHidden[i] += rand.NormFloat64() * 0.2
		}
	}
	// Walk through every output weight and maybe nudge it.
	for i := range a.weightsOutput {
		if rand.Float64() < rate {
			a.weightsOutput[i] += rand.NormFloat64() * 0.2
		}
	}
}

// ---------------------------------------------------------------------
// 3. SIMULATION LOGIC
// ---------------------------------------------------------------------
//
// This section is where the robot actually walks around and gets scored. It is
// the "world" the brain lives in: it turns a position into sensory numbers
// (GetSensoryInputs) and turns a decision into movement (RunMaze).

// GetSensoryInputs builds the 6 numbers the robot "sees" while standing at
// position (x, y). The first 4 numbers are "how far can I walk in this
// direction before hitting a wall", and the last 2 are the robot's own
// coordinates (so the brain can also learn WHERE it is, not just react).
func GetSensoryInputs(x, y int) []float64 {
	inputs := make([]float64, 6) // create the list of 6 sense values

	// dx/dy encode the four directions as step-changes in (row, column):
	//   index 0: Up    (row -1, column  0)
	//   index 1: Down  (row +1, column  0)
	//   index 2: Left  (row  0, column -1)
	//   index 3: Right (row  0, column +1)
	dx := []int{-1, 1, 0, 0}
	dy := []int{0, 0, -1, 1}

	// Do the same "walk and count" for each of the 4 directions.
	for i := 0; i < 4; i++ {
		dist := 0.0    // how many open tiles we could walk before a wall
		cx, cy := x, y // start walking from the robot's current tile

		// This is an infinite loop; we exit it with `break` when we hit a wall
		// or find the exit.
		for {
			// Take one step in the current direction.
			cx += dx[i]
			cy += dy[i]

			// If we stepped onto a wall, stop counting — that's the end of the
			// corridor in this direction.
			if maze[cx][cy] == 1 {
				break
			}

			// Otherwise this tile is walkable, so count it as one tile of
			// visible open space.
			dist += 1.0

			// If this tile is the exit, add a big bonus and stop. This gives
			// the network a strong signal whenever the exit is in sight.
			if maze[cx][cy] == 3 {
				dist += 5.0
				break
			}
		}
		inputs[i] = dist // store the measured distance for this direction
	}

	// The robot also knows its own position (which row and which column it is
	// on). float64(...) just converts the whole numbers into decimal numbers so
	// they match the []float64 list type. This lets the brain learn things like
	// "when I'm in the top-left area, turn right", instead of only reacting to
	// nearby walls.
	inputs[4] = float64(x)
	inputs[5] = float64(y)

	return inputs
}

// RunMaze lets ONE robot try to walk through the maze for at most maxSteps
// moves. It returns true if the robot reached the exit, and it writes the
// robot's "fitness" (score) into agent.fitness either way.
//
// The `*Agent` parameter again means we pass the real robot by reference so we
// can modify its fitness field directly.
func RunMaze(agent *Agent, maxSteps int) bool {
	x, y := startX, startY           // every run begins at the start tile
	minDistToExit := math.MaxFloat64 // remembers the CLOSEST the robot ever got

	// visited remembers every tile the robot has stepped on during this run.
	// It's a map (a dictionary): the key is a text label like "3,4" and the
	// value is how many times that tile was visited. We use it later to reward
	// exploring NEW tiles instead of walking in circles.
	visited := make(map[string]int)

	for step := 0; step < maxSteps; step++ {
		// Record the current tile so we know it has been seen at least once.
		posKey := fmt.Sprintf("%d,%d", x, y) // turn (x,y) into a text key "x,y"
		visited[posKey]++

		// How far (straight-line distance) is the robot from the exit right
		// now? This is the Pythagorean theorem: sqrt(dx^2 + dy^2), where dx and
		// dy are the row and column differences to the exit.
		dx := float64(x - exitX)
		dy := float64(y - exitY)
		dist := math.Sqrt(dx*dx + dy*dy)

		// Remember the smallest distance achieved so far this run.
		if dist < minDistToExit {
			minDistToExit = dist
		}

		// WINNING: if we are standing exactly on the exit, score us and stop.
		if x == exitX && y == exitY {
			// Reward = 1000 minus the number of steps taken. So reaching the
			// exit in FEWER steps gives a HIGHER fitness: faster solutions are
			// favoured by evolution.
			agent.fitness = 1000.0 - float64(step)
			return true
		}

		// Otherwise, ask the robot's brain what to do next.
		inputs := GetSensoryInputs(x, y)
		action := agent.Predict(inputs)

		// Convert the action number into a tentative new position (nx, ny).
		nx, ny := x, y
		switch action {
		case 0:
			nx-- // Up    (row -1)
		case 1:
			nx++ // Down  (row +1)
		case 2:
			ny-- // Left  (column -1)
		case 3:
			ny++ // Right (column +1)
		}

		// If the robot tried to walk INTO a wall, its run ends here (it
		// "crashed"). We do not move it; we simply stop simulating this robot.
		if maze[nx][ny] == 1 {
			break
		}
		x, y = nx, ny // the move is legal, so actually take the step
	}

	// FITNESS for robots that did NOT reach the exit. It has two parts added
	// together:
	//   * (100 / (closestDistance + 1))  rewards getting NEAR the exit. The +1
	//     avoids dividing by zero when the robot starts right on top of it.
	//   * (uniqueTilesExplored * 5)      rewards walking around and seeing NEW
	//     tiles, which discourages spinning in circles.
	// A higher total is better.
	uniqueTilesExplored := float64(len(visited))
	agent.fitness = (100.0 / (minDistToExit + 1.0)) + (uniqueTilesExplored * 5.0)

	return false
}

// ---------------------------------------------------------------------
// 4. MAIN LOOP
// ---------------------------------------------------------------------
//
// This is where EVOLUTION happens. Generation after generation, every robot is
// tested, the best are kept, and mutated copies of them fill the next
// generation — until one robot finally solves the maze.

// main is the entry point of every Go program: it runs automatically when the
// program starts.
func main() {
	// Seed the random number generator with the current time. Without this, Go
	// would produce the SAME random sequence every time the program runs, so
	// evolution would unfold identically each run. Seeding with the current
	// nanoseconds (UnixNano) makes each run different.
	rand.Seed(time.Now().UnixNano())

	populationSize := 100    // how many robots live in each generation
	maxGenerations := 100000 // safety limit so the program can't run forever
	maxSteps := 20           // how many moves each robot gets per test run

	// Build the very first generation: 100 robots with totally random brains.
	population := make([]Agent, populationSize)
	for i := range population {
		population[i] = NewAgent()
	}

	fmt.Println("Starting Evolution...")

	// Loop once per generation (gen starts at 1 and counts upward).
	for gen := 1; gen <= maxGenerations; gen++ {
		success := false    // did ANY robot solve the maze this generation?
		bestFitness := -1.0 // the best score seen so far this generation
		var bestAgent Agent // the robot that achieved that best score

		// Test every robot in the population, one by one.
		for i := range population {
			reachedExit := RunMaze(&population[i], maxSteps)
			if reachedExit {
				success = true
			}
			// Track the single best robot of this generation.
			if population[i].fitness > bestFitness {
				bestFitness = population[i].fitness
				bestAgent = population[i]
			}
		}

		// Print progress so we can watch evolution improve over time.
		// ("%.2f" means "print the number with 2 decimal places".)
		fmt.Printf("Gen %d | Best Fitness: %.2f\n", gen, bestFitness)

		// If a robot solved the maze, celebrate and replay its winning run.
		if success {
			fmt.Printf("✨ Success! An agent solved the maze at Generation %d!\n", gen)
			fmt.Println("Launching real-time visualization in 2 seconds...")
			time.Sleep(2 * time.Second) // give the reader a moment before animating

			// Replay the winning agent visually, then quit the program.
			VisualRun(&bestAgent, maxSteps)
			return
		}

		// --- SELECTION (choose the parents) -------------------------------
		// Build a small group of "survivors": the fittest ~10% of the robots.
		// We pick them with a "tournament": grab two random robots, keep the
		// fitter one, and repeat until we have populationSize/10 survivors.
		survivors := make([]Agent, 0, populationSize/10)
		for len(survivors) < populationSize/10 {
			a1 := population[rand.Intn(populationSize)] // a random robot
			a2 := population[rand.Intn(populationSize)] // another random robot
			if a1.fitness > a2.fitness {
				survivors = append(survivors, a1.Clone()) // a1 wins the duel
			} else {
				survivors = append(survivors, a2.Clone()) // a2 wins (or tie -> a2)
			}
		}

		// --- REPRODUCTION (build the next generation) ---------------------
		// Start the new generation with the single best robot copied as-is.
		// This is called "elitism": we never lose our best solution so far.
		newPopulation := make([]Agent, 0, populationSize)
		newPopulation = append(newPopulation, bestAgent.Clone())

		// Fill the rest of the generation with mutated copies of survivors.
		// Each child is a clone of a random survivor with a 15% chance per
		// weight of being nudged — so children resemble their parents but
		// differ a little, letting evolution explore new possibilities.
		for len(newPopulation) < populationSize {
			parent := survivors[rand.Intn(len(survivors))]
			child := parent.Clone()
			child.Mutate(0.15) // 0.15 = the mutation rate
			newPopulation = append(newPopulation, child)
		}

		// Replace the old generation with the new one, then loop again.
		population = newPopulation
	}

	// Reached only if maxGenerations passes with no solution found.
	fmt.Println("Evolution finished. The maze wasn't completely solved.")
}

// ---------------------------------------------------------------------
// 5. VISUALIZATION LOGIC
// ---------------------------------------------------------------------
//
// This section is purely for US humans: it draws the maze in the terminal so
// we can watch the winning robot walk. It does not affect the learning at all.

// PrintMaze draws the maze into the terminal so we can SEE what is happening.
// agentX/agentY is the tile where the robot currently stands.
func PrintMaze(agentX, agentY int) {
	// "\033[H\033[2J" is an ANSI escape sequence that moves the cursor to the
	// top-left and clears the screen. This makes each new frame overwrite the
	// previous one, creating a simple animation instead of a long scrolling
	// wall of text.
	fmt.Print("\033[H\033[2J")

	// Loop over every row (r) and every column (c) of the maze, printing a
	// symbol for each tile.
	for r := 0; r < len(maze); r++ { // r = row (top to bottom)
		for c := 0; c < len(maze[r]); c++ { // c = column (left to right)
			if r == agentX && c == agentY {
				fmt.Print("🤖 ") // the robot's current position
			} else if maze[r][c] == 1 {
				fmt.Print("██") // a wall
			} else if r == exitX && c == exitY {
				fmt.Print("🚪 ") // the exit door
			} else {
				fmt.Print("  ") // empty path
			}
		}
		fmt.Println() // start a new line after finishing each row
	}
	fmt.Println("\nPress Ctrl+C to exit if it loops.")
}

// VisualRun replays a single robot's journey one step at a time, printing the
// maze after every move so a human can watch the winning robot solve it.
func VisualRun(agent *Agent, maxSteps int) {
	x, y := startX, startY // begin at the start tile

	for step := 0; step < maxSteps; step++ {
		PrintMaze(x, y)
		time.Sleep(200 * time.Millisecond) // pause so the animation is watchable

		// If we are standing on the exit, draw it one final time and celebrate.
		if x == exitX && y == exitY {
			PrintMaze(x, y)
			fmt.Println("\n🎉 Success! The robot reached the exit door!")
			return
		}

		// Ask the brain for its next move and compute the new position.
		inputs := GetSensoryInputs(x, y)
		action := agent.Predict(inputs)

		nx, ny := x, y
		switch action {
		case 0:
			nx-- // Up    (row -1)
		case 1:
			nx++ // Down  (row +1)
		case 2:
			ny-- // Left  (column -1)
		case 3:
			ny++ // Right (column +1)
		}

		// Guard against walking into a wall during the replay (shouldn't happen
		// for a well-trained robot, but it is a safe fallback).
		if maze[nx][ny] == 1 {
			fmt.Println("\n💥 Oops, the winning agent crashed into a wall during replay!")
			return
		}
		x, y = nx, ny // take the step
	}
}
