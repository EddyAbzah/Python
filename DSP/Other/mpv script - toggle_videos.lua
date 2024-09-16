-- Lua script for mpv Video Player to toggle between 3 videos:

-- For file paths you two options: a. path in double quotes but \ is escaped (\\); b. raw string in [[]] but no double quotes.
local video1 = [[Path\01.mp4]]
local video2 = [[Path\01.mp4]]
local video3 = [[Path\01.mp4]]
mp.commandv("loadfile", video1, "replace")
mp.commandv("loadfile", video2, "append")
mp.commandv("loadfile", video3, "append")

local delay = 0.05
local current_video = 1
local time_pos
local video_path


function toggle_videos()
    time_pos = mp.get_property_number("time-pos")
    if current_video == 1 then
        get_video_2()
    elseif current_video == 2 then
        get_video_3()
	else
        get_video_1()
    end
	mp.add_timeout(delay, function()
        mp.set_property_number("time-pos", time_pos)
    end)
end

function get_video_1()
    current_video = 1
    mp.commandv("loadfile", video1, "replace")
end

function get_video_2()
    current_video = 2
    mp.commandv("loadfile", video2, "replace")
end

function get_video_3()
    current_video = 3
    mp.commandv("loadfile", video3, "replace")
end

mp.add_key_binding("TAB", "toggle-videos", toggle_videos)
mp.add_key_binding("F1", "get_video_1", get_video_1)
mp.add_key_binding("F2", "get_video_2", get_video_2)
mp.add_key_binding("F3", "get_video_3", get_video_3)
