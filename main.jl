using Distributions

const schema = [1, 1]
const parameterized_layers_length = length(schema) - 1
const learning_rate = 0.1

a = Vector(undef, parameterized_layers_length)
z = Vector(undef, parameterized_layers_length)
w = Vector(undef, parameterized_layers_length)
b = Vector(undef, parameterized_layers_length)

for l = 1:parameterized_layers_length
    unif = Uniform(-1, 1)
    w[l] = rand(unif, schema[l+1], schema[l])
    b[l] = rand(unif, schema[l+1], 1)
end

function cost(output, target)
    error = output - target
    errorSquared = map((x) -> x^2, error)
    return reduce(+, errorSquared)
end
function cost_der(output, target)
    error = output - target
    derivative = map((x) -> 2x, error)
    return reduce(+, derivative)
end

function σ(x)
    return 1 / (1 + exp(-x))
end
function σ_der(x)
    return σ(x) * (1 - σ(x))
end

""" Actually called hadamard product """
function not_dot_multiply(a, b)
    if length(a) !== length(b)
        error("Length of a and b must be equal")
    end
    c = Vector(undef, length(a))
    for i = 1:length(a)
        c[i] = a[i] * b[i]
    end
    return c
end

function feed_forward(input)
    previous_a = input
    for l = 1:parameterized_layers_length
        z[l] = w[l] * previous_a + b[l]
        a[l] = map(σ, z[l])
        previous_a = a[l]
    end
    return a[parameterized_layers_length]
end

function back_propagate(input, output, target)
    ∂c_a = cost_der(output, target)
    for l = parameterized_layers_length:-1:1
        ∂z_w = l - 1 == 0 ? input : a[l-1]
        ∂σ_z = map(σ_der, z[l])

        if l == parameterized_layers_length
            ∂c_w = (∂c_a * ∂σ_z) * transpose(∂z_w)
            ∂c_b = (∂c_a * ∂σ_z)
        else
            ∂c_w = not_dot_multiply(∂c_a, ∂σ_z) * transpose(∂z_w)
            ∂c_b = not_dot_multiply(∂c_a, ∂σ_z)
        end

        chain = isa(∂c_a, Number) ?
                (∂c_a * ∂σ_z) : not_dot_multiply(∂c_a, ∂σ_z)
        ∂c_a = transpose(w[l]) * chain
        w[l] = w[l] - ∂c_w * learning_rate
        b[l] = b[l] - ∂c_b * learning_rate
    end
end

function choice(a)
    highest = a[1]
    highest_index = 1
    for i = 2:length(a)
        if a[i] > highest
            highest = a[i]
            highest_index = i
        end
    end
    return [highest_index, highest]
end

function train(x, y, epochs=100)
    length_x = length(x)
    if length_x !== length(y)
        error("Length of x and y must be equal")
    end
    io = open("res.txt", "w")
    for epoch = 1:epochs
        correct = 0
        println("starting epoch ", epoch)
        for i = 1:length_x
            output = feed_forward(x[i])
            d1 = choice(output)[1]
            d2 = choice(y[i])[1]
            write(io, "[$d1 $d2]\n")
            if choice(output)[1] == choice(y[i])[1]
                correct += 1
            end
            print(output, y[i])
            back_propagate(x[i], output, y[i])
        end
        println("ending epoch ", epoch,
            " with accuracy ", correct / length_x)
    end
    close(io)
end

function test(x, y)
    if length(x) !== length(y)
        error("Length of x and y must be equal")
    end
    correct = 0
    for i = 1:length(x)
        output = feed_forward(x[i])
        if choice(output)[1] == choice(y[i])[1]
            correct += 1
        end
    end

    return correct / length(x)
end