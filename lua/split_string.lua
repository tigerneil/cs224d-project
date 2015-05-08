
local M = {}

-- Splits a string similar to python's str.split().
-- Source: http://stackoverflow.com/questions/1426954/split-string-in-lua
function M.split(inputstr, sep)
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

return M