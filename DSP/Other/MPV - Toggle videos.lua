local file_list = {
    [[]],
    [[]]
}


local delay = 0.05
local last_position = 0
mp.set_property("osd-font-size", 20)


local function load_file(index)
    local file = file_list[index]
    if file then
		mp.osd_message(file, 2)
		last_position = mp.get_property_number("time-pos") or 0
        mp.commandv("loadfile", file, "replace")
		mp.add_timeout(delay, function()
			mp.set_property_number("time-pos", last_position)
		end)
    else
		mp.osd_message("Invalid index: " .. index, 2)
    end
end


for i = 1, #file_list do
    local key = "F" .. i
    mp.add_key_binding(key, "load_file_" .. key, function()
        load_file(i)
    end)
end
