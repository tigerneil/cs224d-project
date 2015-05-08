-- Helpful utilities in Lua.

local util = {}

function util.setDefault(t, d)
        local mt = {__index = function () return d end}
        setmetatable(t, mt)
end


-- Splits a string similar to python's str.split().
-- Source: http://stackoverflow.com/questions/1426954/split-string-in-lua
function util.split(inputstr, sep)
		if sep == "" then
			return {}
		end
        if sep == nil then
                sep = "%s"
        end
        local t={} ; i=1
        for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
                t[i] = str
                i = i + 1
        end
        return t
end

-- File IO utility
function util.readDataFile(path, processLine)
    local path = path
    print "in read"
    local inputFile = io.open(path)
    
    local line = inputFile:read("*l")
    while line do
       processLine(line)
       line = inputFile:read("*l")
    end
end

return util