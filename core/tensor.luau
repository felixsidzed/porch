--!strict
--!optimize 2

-- export type Tensor = {
-- 	data: buffer,
-- 	shape: { number },
-- 	stride: { number },
-- 	size: number,
-- }

local Tensor = {}
Tensor.__index = Tensor

export type Tensor = typeof(setmetatable(
	{} :: {
		data: any,
		shape: { number },
		stride: { number },
		size: number,
	},
	Tensor
))

local function numel(shape: { number }): number
	local n = 1
	for i = 1, #shape do
		n *= shape[i]
	end
	return n
end

local function computeStrides(shape: { number }): { number }
	local strides = table.create(#shape)
	local stride = 1
	for i = #shape, 1, -1 do
		strides[i] = stride
		stride *= shape[i]
	end
	return strides
end

local function censorflow(buf, shape, stride, size): Tensor -- w tensorflow stray -- pytorch better
	return (setmetatable({data=buf, shape=shape, stride=stride, size=size}, Tensor) :: any) :: Tensor
end

-- CREATION

function Tensor.new(data: { number }, shape: { number }): Tensor
	local size = numel(shape)
	assert(#data == size, "#data does not match shape")

	local buf = buffer.create(size * 4)

	for i = 1, size do
		buffer.writef32(buf, (i - 1) * 4, data[i])
	end

	return censorflow(buf, shape, computeStrides(shape), size)
end

function Tensor.zeros(shape: { number }): Tensor
	local size = numel(shape)
	local buf = buffer.create(size * 4)
	buffer.fill(buf, 0, 0, size * 4)

	return censorflow(buf, shape, computeStrides(shape), size)
end

function Tensor.ones(shape: { number }): Tensor
	local size = numel(shape)
	local buf = buffer.create(size * 4)

	for i = 0, size - 1 do
		buffer.writef32(buf, i * 4, 1)
	end

	return censorflow(buf, shape, computeStrides(shape), size)
end

function Tensor.randn(shape: { number }): Tensor
	local size = numel(shape)
	local buf = buffer.create(size * 4)

	for i = 0, size - 1 do
		buffer.writef32(buf, i * 4, (math.random() - 0.5) * 2)
	end

	return censorflow(buf, shape, computeStrides(shape), size)
end

-- SHAPING

function Tensor:reshape(newShape: { number }): Tensor
	assert(numel(newShape) == self.size, "invalid reshape")
	return censorflow(self.data, newShape, computeStrides(newShape), self.size)
end

function Tensor:flatten(): Tensor
	return (self :: any):reshape({ self.size })
end

function Tensor:transpose(...)
	local shape = {table.unpack(self.shape)}
	local stride = {table.unpack(self.stride)}

	local dims = {...}
	assert(#dims % 2 == 0, "... must be pairs")

	for i = 1, #dims, 2 do
		local a, b = dims[i], dims[i+1]
		shape[a], shape[b] = shape[b], shape[a]
		stride[a], stride[b] = stride[b], stride[a]
	end

	return censorflow(self.data, shape, stride, self.size)
end

function Tensor:slice(dim: number, start: number, stop: number): Tensor
	local newShape = table.clone(self.shape)
	newShape[dim] = stop - start + 1

	local newSize = numel(newShape)
	local buf = buffer.create(newSize * 4)

	local idx = 0
	for i = start, stop do
		local off = (i - 1) * self.stride[dim] * 4
		buffer.copy(buf, idx * 4, self.data, off, 4)
		idx += 1
	end

	return censorflow(buf, newShape, computeStrides(newShape), newSize)
end

-- MATH OPS

function Tensor.__add(a: Tensor, b: Tensor): Tensor
	assert(a.size == b.size)

	local out = buffer.create(a.size * 4)
	for i = 0, a.size - 1 do
		local v = buffer.readf32(a.data, i * 4) + buffer.readf32(b.data, i * 4)
		buffer.writef32(out, i * 4, v)
	end
	return censorflow(out, a.shape, a.stride, a.size)
end

function Tensor.__sub(a: Tensor, b: Tensor): Tensor
	assert(a.size == b.size)

	local out = buffer.create(a.size * 4)
	for i = 0, a.size - 1 do
		local v = buffer.readf32(a.data, i * 4) - buffer.readf32(b.data, i * 4)
		buffer.writef32(out, i * 4, v)
	end
	return censorflow(out, a.shape, a.stride, a.size)
end

function Tensor.__mul(a: Tensor, b: Tensor): Tensor
	assert(a.size == b.size)

	local out = buffer.create(a.size * 4)
	for i = 0, a.size - 1 do
		local v = buffer.readf32(a.data, i * 4) * buffer.readf32(b.data, i * 4)
		buffer.writef32(out, i * 4, v)
	end
	return censorflow(out, a.shape, a.stride, a.size)
end

function Tensor.__div(a: Tensor, b: Tensor): Tensor
	assert(a.size == b.size)

	local out = buffer.create(a.size * 4)
	for i = 0, a.size - 1 do
		local v = buffer.readf32(a.data, i * 4) / buffer.readf32(b.data, i * 4)
		buffer.writef32(out, i * 4, v)
	end
	return censorflow(out, a.shape, a.stride, a.size)
end

function Tensor.__pow(a: Tensor, b: Tensor): Tensor
	assert(a.size == b.size)

	local out = buffer.create(a.size * 4)
	for i = 0, a.size - 1 do
		local v = buffer.readf32(a.data, i * 4) ^ buffer.readf32(b.data, i * 4)
		buffer.writef32(out, i * 4, v)
	end
	return censorflow(out, a.shape, a.stride, a.size)
end

function Tensor:add(other: Tensor): Tensor
	assert(self.size == other.size)

	for i = 0, (self.size :: number) - 1 do
		local v = buffer.readf32(self.data, i * 4) + buffer.readf32(other.data, i * 4)
		buffer.writef32(self.data, i * 4, v)
	end
	return self
end

function Tensor:sub(other: Tensor): Tensor
	assert(self.size == other.size)

	for i = 0, (self.size :: number) - 1 do
		local v = buffer.readf32(self.data, i * 4) - buffer.readf32(other.data, i * 4)
		buffer.writef32(self.data, i * 4, v)
	end
	return self
end

function Tensor:mul(other: Tensor): Tensor
	assert(self.size == other.size)

	for i = 0, (self.size :: number) - 1 do
		local v = buffer.readf32(self.data, i * 4) * buffer.readf32(other.data, i * 4)
		buffer.writef32(self.data, i * 4, v)
	end
	return self
end

function Tensor:div(other: Tensor): Tensor
	assert(self.size == other.size)

	for i = 0, (self.size :: number) - 1 do
		local v = buffer.readf32(self.data, i * 4) / buffer.readf32(other.data, i * 4)
		buffer.writef32(self.data, i * 4, v)
	end
	return self
end

function Tensor:pow(other: Tensor): Tensor
	assert(self.size == other.size)

	for i = 0, (self.size :: number) - 1 do
		local v = buffer.readf32(self.data, i * 4) ^ buffer.readf32(other.data, i * 4)
		buffer.writef32(self.data, i * 4, v)
	end
	return self
end

function Tensor.sum(t: Tensor, dim: number?): number | Tensor
	if dim == nil then
		local s = 0
		for i = 0, t.size - 1 do
			s += buffer.readf32(t.data, i * 4)
		end
		return s
	end

	error("NotImplemented")
end

function Tensor.mean(t: Tensor, dim: number?): number | Tensor
	if dim == nil then
		return (Tensor.sum(t, nil) :: number) / t.size
	end
	error("NotImplemented")
end

function Tensor.max(t: Tensor, dim: number?): number | Tensor
	if dim == nil then
		local m = -math.huge
		for i = 0, t.size - 1 do
			m = math.max(m, buffer.readf32(t.data, i * 4))
		end
		return m
	end
	error("NotImplemented")
end

function Tensor.matmul(a: Tensor, b: Tensor): Tensor
	local arows = a.shape[1]
	local acols = a.shape[2]
	local bcols = b.shape[2]

	assert(acols == b.shape[1])

	local out = buffer.create(arows * bcols * 4)
	for i = 1, arows do
		for j = 1, bcols do
			local sum = 0
			for k = 1, acols do
				local aIdx = ((i - 1) * acols + (k - 1)) * 4
				local bIdx = ((k - 1) * bcols + (j - 1)) * 4

				sum += buffer.readf32(a.data, aIdx) * buffer.readf32(b.data, bIdx)
			end

			buffer.writef32(out, ((i - 1) * bcols + (j - 1)) * 4, sum)
		end
	end

	local shape = { arows, bcols }
	return censorflow(out, shape, computeStrides(shape), arows * bcols)
end

function Tensor:dot(a: Tensor, b: Tensor): number
	assert(a.size == b.size)

	local sum = 0
	for i = 0, a.size - 1 do
		sum += buffer.readf32(a.data, i * 4) * buffer.readf32(b.data, i * 4)
	end

	return sum
end

-- UTIKL
function Tensor:__tostring()
	if #self.shape == 1 then
		local row = {}
		for i = 0, self.shape[1] - 1 do
			row[#row + 1] = tostring(buffer.readf32(self.data, i * 4))
		end
		return "{ " .. table.concat(row, ", ") .. " }"
	end

	local rows = {}
	local row_size = self.shape[#self.shape]

	for r = 0, (self.size / row_size) - 1 do
		local row = {}
		for c = 0, row_size - 1 do
			local idx = r * row_size + c
			row[#row + 1] = tostring(buffer.readf32(self.data, idx * 4))
		end
		rows[#rows + 1] = "{ " .. table.concat(row, ", ") .. " }"
	end

	return "{ " .. table.concat(rows, ",\n  ") .. " }"
end

return Tensor
