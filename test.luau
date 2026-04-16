--!strict
--!optimize 2

local Tensor = require("./tensor")

local a = Tensor.new(
	{1, 2, 3,
	 4, 5, 6},
	{3, 2}
)

local b = Tensor.new(
	{10, 20, 30,
	 40, 50, 60},
	{3, 2}
)

print(a + b)
print(a * b)

print(a)
print(a:matmul(b:reshape({ 2, 3 })))
