set file_a [lindex $argv 0]
set file_b [lindex $argv 1]
set file_c [lindex $argv 2]

source remove_long_bonds.tcl

mol addrep 0
display resetview
mol new $file_a type {psf} first 0 last -1 step 1 waitfor 1
animate style Loop
display resetview
mol addfile $file_b type {lammpstrj} first 0 last -1 step 1 waitfor -1
animate style Loop

color Display Background white
pbc box -color black -center origin -width 2 -material AOChalky
rotate y by 90
rotate x by 20
remove_long_bonds 18

axes location off

color Name 1 blue2
color Name 2 orange2

mol modmaterial 0 0 AOChalky
material change outline AOChalky 4.0
mol modstyle 0 0 Licorice 0.5 100.0 100.0

display shadows on
display ambientocclusion on
display aoambient 1.0
display aodirect 0.4
display height 5.0
display projection Orthographic
#display rendermode GLSL

light 0 on
light 1 on
#light 2 on
#light 3 on

#scale by 0.83
#scale by 0.83
#scale by 0.83

render Tachyon $file_c "/usr/local/lib/vmd/tachyon_LINUXAMD64" -aasamples 12 %s -format TARGA -res 3000 1500 -o %s.tga

mol modstyle 0 0 QuickSurf 0.5 0.7 0.5 3.0
mol modselect 0 0 name 2

render Tachyon $file_c.surface "/usr/local/lib/vmd/tachyon_LINUXAMD64" -aasamples 12 %s -format TARGA -res 3000 1500 -o %s.tga
