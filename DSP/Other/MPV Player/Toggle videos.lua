-- IMPORTANT
-- To update the bind keys to the current playlist, press "Shift+F1"



-- Initial settings:
mp.set_property("osd-font-size", 20)
local delay = 0.05						-- delay to set the time position correctly 
local last_position = 0					-- to save the time position of the current file
local set_intial_playlist = false		-- to set the playlist at stratup
local initial_bind_count = 8			-- bind this many keys (so you wouldn't have to reset the bind keys)



-- Initial files to load:
local prefix = [[]]
local suffix = [[]]
local file_list = {
    prefix .. [[]] .. suffix,
    prefix .. [[]] .. suffix,
}


-- Load files:
local function load_file(index)
	if index <= #file_list then				-- Lua uses 1-based indexing
		local file = file_list[index]
		if #file > 0 then					-- # = length
			last_position = mp.get_property_number("time-pos") or 0
			mp.commandv("playlist-play-index", index - 1)
			mp.osd_message(file, 2)
			mp.add_timeout(delay, function()
				mp.set_property_number("time-pos", last_position)
			end)
		else
			mp.osd_message("No file in playlist for key: F" .. index .. "\nPress Shift+F1 to update the playlist", 4)
		end
	else
		mp.osd_message("No file in playlist for key: F" .. index .. "\nPress Shift+F1 to update the playlist", 4)
	end
end


-- Bind keys for loading files:
local function bind_keys(bind_keys_count)
    for i = 1, bind_keys_count do
        local key = "F" .. i
        mp.add_key_binding(key, "load_file_" .. key, function()
            load_file(i)
        end)
    end
	mp.osd_message("Playlist updated. Items: " .. #file_list, 2)
end


-- Function to update file_list from the current MPV playlist:
local function update_file_list()
    local playlist = mp.get_property_native("playlist") or {}
    file_list = {}
    for _, item in ipairs(playlist) do
        table.insert(file_list, item.filename)
    end
	bind_keys(#file_list)
end


-- Set the initial playlist without playing
local function initial_playlist()
	for i, file in ipairs(file_list) do
		if file and string.len(file) > 0 then
			mp.commandv("loadfile", file, "append")
		end
	end
	bind_keys(#file_list)
end


-- Initial setup:
if set_intial_playlist then
	initial_playlist()
else
    bind_keys(initial_bind_count)
end
mp.add_key_binding("Shift+F1", "update_file_list", update_file_list)
update_file_list()
