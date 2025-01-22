-- Initial settings:
mp.set_property("osd-font-size", 20)
local delay = 0.05
local last_position = 0


-- Initial files to load:
local prefix = [[]]
local suffix = [[]]
local file_list = {
    prefix .. [[]] .. suffix,
    prefix .. [[]] .. suffix,
}


-- Load files:
local function load_file(index)
    local file = file_list[index]
    if file then
        mp.osd_message(file, 2)
        last_position = mp.get_property_number("time-pos") or 0
        --mp.commandv("loadfile", file, "replace")
        mp.commandv("playlist-play-index", index - 1)
        mp.add_timeout(delay, function()
            mp.set_property_number("time-pos", last_position)
        end)
    else
        mp.osd_message("Invalid index: " .. index, 2)
    end
end


-- Bind keys for loading files:
local function bind_keys()
    for i = 1, #file_list do
        local key = "F" .. i
        mp.add_key_binding(key, "load_file_" .. key, function()
            load_file(i)
        end)
    end
end


-- Function to update file_list from the current MPV playlist:
local function update_file_list()
    local playlist = mp.get_property_native("playlist") or {}
    file_list = {}
    for _, item in ipairs(playlist) do
        table.insert(file_list, item.filename)
    end
	bind_keys()
    mp.osd_message("Playlist updated. Items: " .. #file_list, 2)
end


-- Set the initial playlist without playing
for i, file in ipairs(file_list) do
	if file and string.len(file) > 0 then
		mp.commandv("loadfile", file, "append")
	end
end


-- Initial setup:
bind_keys()
mp.add_key_binding("Shift+F1", "update_file_list", update_file_list)
