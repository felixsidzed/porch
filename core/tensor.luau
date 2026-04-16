--!strict
--!optimize 2

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
	return setmetatable({data=buf, shape=shape, stride=stride, size=size}, Tensor)
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

	-- outdated runtimes line lune dont have tau
	local tau = math.tau or math.pi * 2
    
    local i = 0
    while i < size do
        local u1 = math.max(math.random(), 1e-10)
        local u2 = math.random()
        local mag = math.sqrt(-2 * math.log(u1))
        buffer.writef32(buf, i * 4, mag * math.cos(tau * u2))
        if i + 1 < size then
            buffer.writef32(buf, (i + 1) * 4, mag * math.sin(tau * u2))
        end
        i += 2
    end

    return censorflow(buf, shape, computeStrides(shape), size)
end

-- SHAPING

function Tensor:reshape(newShape: { number }): Tensor
	assert(numel(newShape) == self.size, "invalid reshape")
	return censorflow(self.data, newShape, computeStrides(newShape), self.size)
end

function Tensor:flatten(): Tensor
	return (self :: any):reshape(self.size)
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
-- dear luau-lsp this type checking is booty do better

local function offset(self: Tensor, linear: number)
	local idx = 0
	for k = 1, #self.shape do
		local s = self.stride[k]
		idx += math.floor(linear / s) % self.shape[k] * s
	end
	return idx * 4
end

function Tensor.__add(a: Tensor, b): Tensor
	local isScalar = type(b) == "number"
	assert(isScalar or (a.size % b.size) == 0)

	local out = buffer.create(a.size * 4)
	for i = 0, a.size - 1 do
		local v; if isScalar then v = b
		else v = buffer.readf32(b.data, offset(b, i % b.size)) end
		local off = offset(a, i)
		buffer.writef32(out, off, buffer.readf32(a.data, off) + v)
	end
	return censorflow(out, table.clone(a.shape), computeStrides(a.shape), a.size)
end

function Tensor.__sub(a: Tensor, b): Tensor
	local isScalar = type(b) == "number"
	assert(isScalar or (a.size % b.size) == 0)

	local out = buffer.create(a.size * 4)
	for i = 0, a.size - 1 do
		local v; if isScalar then v = b
		else v = buffer.readf32(b.data, offset(b, i % b.size)) end
		local off = offset(a, i)
		buffer.writef32(out, off, buffer.readf32(a.data, off) - v)
	end
	return censorflow(out, table.clone(a.shape), computeStrides(a.shape), a.size)
end

function Tensor.__mul(a: Tensor, b): Tensor
	local isScalar = type(b) == "number"
	assert(isScalar or (a.size % b.size) == 0)

	local out = buffer.create(a.size * 4)
	for i = 0, a.size - 1 do
		local v; if isScalar then v = b
		else v = buffer.readf32(b.data, offset(b, i % b.size)) end
		local off = offset(a, i)
		buffer.writef32(out, off, buffer.readf32(a.data, off) * v)
	end
	return censorflow(out, table.clone(a.shape), computeStrides(a.shape), a.size)
end

function Tensor.__div(a: Tensor, b): Tensor
	local isScalar = type(b) == "number"
	assert(isScalar or (a.size % b.size) == 0)

	local out = buffer.create(a.size * 4)
	for i = 0, a.size - 1 do
		local v; if isScalar then v = b
		else v = buffer.readf32(b.data, offset(b, i % b.size)) end
		local off = offset(a, i)
		buffer.writef32(out, off, buffer.readf32(a.data, off) / v)
	end
	return censorflow(out, table.clone(a.shape), computeStrides(a.shape), a.size)
end

function Tensor.__pow(a: Tensor, b): Tensor
	local isScalar = type(b) == "number"
	assert(isScalar or (a.size % b.size) == 0)

	local out = buffer.create(a.size * 4)
	for i = 0, a.size - 1 do
		local v; if isScalar then v = b
		else v = buffer.readf32(b.data, offset(b, i % b.size)) end
		local off = offset(a, i)
		buffer.writef32(out, off, buffer.readf32(a.data, off) ^ v)
	end
	return censorflow(out, table.clone(a.shape), computeStrides(a.shape), a.size)
end

function Tensor:add(other): Tensor
	local isScalar = type(other) == "number"
	if not isScalar and self.size % other.size ~= 0 then
		print(debug.traceback(nil, 2))
	end
	assert(isScalar or (self.size % other.size) == 0)

	for i = 0, (self.size :: number) - 1 do
		local v; if isScalar then v = other
		else v = buffer.readf32(other.data, offset(other, i % other.size)) end
		local off = offset(self, i)
		buffer.writef32(self.data, off, buffer.readf32(self.data, off) + v)
	end
	return self
end

function Tensor:sub(other): Tensor
	local isScalar = type(other) == "number"
	assert(isScalar or (self.size % other.size) == 0)

	for i = 0, (self.size :: number) - 1 do
		local v; if isScalar then v = other
		else v = buffer.readf32(other.data, offset(other, i % other.size)) end
		local off = offset(self, i)
		buffer.writef32(self.data, off, buffer.readf32(self.data, off) - v)
	end
	return self
end

function Tensor:mul(other): Tensor
	local isScalar = type(other) == "number"
	assert(isScalar or (self.size % other.size) == 0)

	for i = 0, (self.size :: number) - 1 do
		local v; if isScalar then v = other
		else v = buffer.readf32(other.data, offset(other, i % other.size)) end
		local off = offset(self, i)
		buffer.writef32(self.data, off, buffer.readf32(self.data, off) * v)
	end
	return self
end

function Tensor:div(other): Tensor
	local isScalar = type(other) == "number"
	assert(isScalar or (self.size % other.size) == 0)

	for i = 0, (self.size :: number) - 1 do
		local v; if isScalar then v = other
		else v = buffer.readf32(other.data, offset(other, i % other.size)) end
		local off = offset(self, i)
		buffer.writef32(self.data, off, buffer.readf32(self.data, off) / v)
	end
	return self
end

function Tensor:pow(other): Tensor
	local isScalar = type(other) == "number"
	assert(isScalar or (self.size % other.size) == 0)

	for i = 0, (self.size :: number) - 1 do
		local v; if isScalar then v = other
		else v = buffer.readf32(other.data, offset(other, i % other.size)) end
		local off = offset(self, i)
		buffer.writef32(self.data, off, buffer.readf32(self.data, off) ^ v)
	end
	return self
end

function Tensor:sum(dim: number?): number | Tensor
	if dim == nil then
		local s = 0
		for i = 0, self.size - 1 do
			s += buffer.readf32(self.data, offset(self, i))
		end
		return s
	elseif dim == 0 then
		local rows, cols = self.shape[1], self.shape[2]

		local out = Tensor.zeros({cols})
		for c = 0, cols - 1 do
			local s = 0
			for r = 0, rows - 1 do
				s += buffer.readf32(self.data, offset(self, r * cols + c))
			end
			buffer.writef32(out.data, offset(out, c), s)
		end

		return out
	end

	error("NotImplemented")
end

function Tensor:mean(dim: number?): number | Tensor
	if dim == nil then
		return (self:sum(nil) :: number) / self.size
	end
	return (self:sum(dim) :: Tensor) / self.shape[dim]
end

function Tensor:max(dim: number?): number | Tensor
	if dim == nil then
		local m = -math.huge
		for i = 0, self.size - 1 do
			m = math.max(m, buffer.readf32(self.data, offset(self, i)))
		end
		return m
	end
	error("NotImplemented")
end

local function lucybedrock(v0: { number }, v1, v2)
		local v3 = 0
		for v4 = 1, #v0 do v3 += v0[v4] * v2[v4] end
		return v3 * 4
	end

function Tensor.matmul(a: Tensor, b: Tensor)
	local r, c1, c2 = a.shape[1], a.shape[2], b.shape[2]
	assert(c1 == b.shape[1])

	local out = buffer.create(r * c2 * 4)
	local sh, st = {r, c2}, computeStrides({r, c2})

	for i = 0, r-1 do
		for j = 0, c2-1 do
			local sum = 0
			for k = 0, c1-1 do
				sum +=	buffer.readf32(a.data, lucybedrock({i,k}, a.shape, a.stride)) *
						buffer.readf32(b.data, lucybedrock({k,j}, b.shape, b.stride))
			end
			buffer.writef32(out, lucybedrock({i,j}, sh, st), sum)
		end
	end

	return censorflow(out, sh, st, r * c2)
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
	local rows = {}
	local rowSize = self.shape[#self.shape]

	for r = 0, (self.size / rowSize) - 1 do
		local row = {}
		for c = 0, rowSize - 1 do
			table.insert(row, buffer.readf32(self.data, (r * rowSize + c) * 4))
		end
		table.insert(rows, `\{ {table.concat(row, ", ")} }`)
	end

	return `\{ {table.clone(rows, ",\n ")} }`
end

return Tensor
