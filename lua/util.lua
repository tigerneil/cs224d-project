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

-- Syntax for reading a file. Can't actually use!
-- function util.readDataFile(path, func)
--     local path = path
--     print "in read"
--     local inputFile = io.open(path)
    
--     local line = inputFile:read("*l")
--     while line do
--        func(line)
--        line = inputFile:read("*l")
--     end
-- end


-- Parse a line form a python-processed file.
function util.parseProcessedLine(line)
    -- split the line by tab
    vals = util.split(line, "\t")
    tokens = util.split(vals[1], ",")
    relations = util.split(vals[2], ",")
    return tokens, relations
end

return util