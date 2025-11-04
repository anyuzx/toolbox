# VMD Script to color structure by binary values (0 or 1)
# Usage: vmd -e vmd_color_binary.tcl -args <xyz_file> <binary_data_file>
# Or within VMD: source vmd_color_binary.tcl <xyz_file> <binary_data_file>

# ============================================
# COLOR CONFIGURATION - Modify these colors as needed
# ============================================
# Available VMD colors: blue, red, gray, orange, yellow, tan, silver,
# green, white, pink, cyan, purple, lime, mauve, ochre, iceblue, black,
# yellow2, yellow3, green2, green3, cyan2, cyan3, blue2, blue3, violet,
# violet2, magenta, magenta2, red2, red3, orange2, orange3
set color_type_0 "blue2"     ;# Color for state 0
set color_type_1 "orange2"      ;# Color for state 1

# ============================================
# Parse command-line arguments
# ============================================
if {$argc < 1} {
    puts "ERROR: Missing required arguments"
    puts "Usage: vmd -e vmd_color_binary.tcl -args <xyz_file> [binary_data_file]"
    puts "  <xyz_file>         : Path to XYZ structure file (required)"
    puts "  <binary_data_file> : Path to text file containing 0 or 1 values (optional)"
    exit
}

set xyz_file [lindex $argv 0]
set binary_data_file ""
set use_binary_coloring 0

if {$argc >= 2} {
    set binary_data_file [lindex $argv 1]
    set use_binary_coloring 1
}

puts "========================================"
puts "Loading structure from: $xyz_file"
if {$use_binary_coloring} {
    puts "Loading binary data from: $binary_data_file"
} else {
    puts "No binary data file provided - using default coloring"
}
puts "========================================"

# Check if files exist
if {![file exists $xyz_file]} {
    puts "ERROR: XYZ file not found: $xyz_file"
    exit
}
if {$use_binary_coloring && ![file exists $binary_data_file]} {
    puts "ERROR: Binary data file not found: $binary_data_file"
    exit
}

# ============================================
# Load structure
# ============================================
mol new $xyz_file type xyz first 0 last -1 step 1 filebonds 0 autobonds 0 waitfor all
set molid [molinfo top]

# ============================================
# Read binary data (0 or 1) - only if provided
# ============================================
if {$use_binary_coloring} {
    set fp [open $binary_data_file r]
    set binary_values {}
    while {[gets $fp line] >= 0} {
        set line [string trim $line]
        # Skip empty lines and comments
        if {$line != "" && [string index $line 0] != "#"} {
            set value [string trim $line]
            # Validate that value is 0 or 1
            if {$value != "0" && $value != "1"} {
                puts "WARNING: Found non-binary value '$value', treating as 0"
                set value 0
            }
            lappend binary_values $value
        }
    }
    close $fp

    set n_values [llength $binary_values]
    puts "Loaded $n_values binary values from $binary_data_file"

    # Count 0s and 1s
    set count_0 0
    set count_1 0
    foreach val $binary_values {
        if {$val == 0} {
            incr count_0
        } else {
            incr count_1
        }
    }
    puts "  Type 0: $count_0 atoms"
    puts "  Type 1: $count_1 atoms"

    # ============================================
    # Set up atom selection and assign values
    # ============================================
    set sel [atomselect $molid "all"]
    set n_atoms [$sel num]

    if {$n_atoms != $n_values} {
        puts "WARNING: Number of atoms ($n_atoms) does not match number of binary values ($n_values)"
        if {$n_atoms > $n_values} {
            puts "  Padding with 0s for remaining atoms"
            for {set i $n_values} {$i < $n_atoms} {incr i} {
                lappend binary_values 0
            }
        } else {
            puts "  Truncating binary values to match number of atoms"
            set binary_values [lrange $binary_values 0 [expr $n_atoms - 1]]
        }
    }

    # Assign binary values to the 'user' field (for coloring)
    # We use 'user' field as it's a float field that can be used for coloring
    $sel set user $binary_values

    # Also assign to 'type' field for reference
    # Type 0 -> "A", Type 1 -> "B"
    set type_names {}
    foreach val $binary_values {
        if {$val == 0} {
            lappend type_names "A"
        } else {
            lappend type_names "B"
        }
    }
    $sel set type $type_names
} else {
    # No binary coloring - just set up selection
    set sel [atomselect $molid "all"]
    set n_atoms [$sel num]
    puts "No binary coloring applied"
}

# ============================================
# Build explicit chain bonds: 1-2, 2-3, ..., (N-1)-N
# ============================================
if {[catch {package require topotools}]} {
    puts "WARNING: topotools not found; cannot add bonds."
    puts "  Install topotools or bonds may not display correctly"
} else {
    topo clearbonds
    for {set i 0} {$i < [expr {$n_atoms - 1}]} {incr i} {
        topo addbond $i [expr {$i + 1}]
    }
    puts "Added chain bonds: $n_atoms atoms -> [expr {$n_atoms - 1}] bonds"
}

# ============================================
# Assign types to atoms based on binary values  
# ============================================
if {$use_binary_coloring} {
    # Get color IDs from color names (use 'index' not 'num')
    set colorid_0 [colorinfo index $color_type_0]
    set colorid_1 [colorinfo index $color_type_1]

    puts "Color scheme:"
    puts "  State 0 (Type A) -> $color_type_0 (ID: $colorid_0)"
    puts "  State 1 (Type B) -> $color_type_1 (ID: $colorid_1)"

    # Assign atom types based on binary values: 0 -> "A", 1 -> "B"
    set type_names {}
    foreach val $binary_values {
        if {$val == 0} {
            lappend type_names "A"
        } else {
            lappend type_names "B"
        }
    }
    $sel set type $type_names

    # Also assign to name field as backup
    $sel set name $type_names

    puts "Assigned atom types: A (state 0) and B (state 1)"

    # Set the colors for atom types A and B
    color Type A $color_type_0
    color Type B $color_type_1

    puts "Set Type colors: A=$color_type_0, B=$color_type_1"
} else {
    puts "Using default atom coloring (no binary data)"
}

# ============================================
# Customize AOShiny material properties
# ============================================
# Modify these values as needed
set material_ambient 0.2      ;# Ambient light contribution (0.0-1.0)
set material_diffuse 0.7      ;# Diffuse light contribution (0.0-1.0)
set material_outline 1.0      ;# Outline width (0.0-6.0, 0 = no outline)

# Apply material property changes
material change ambient AOShiny $material_ambient
material change diffuse AOShiny $material_diffuse
material change outline AOShiny $material_outline

puts "AOShiny material properties:"
puts "  Ambient:  $material_ambient"
puts "  Diffuse:  $material_diffuse"
puts "  Outline:  $material_outline"

# ============================================
# Visualization settings - Single Licorice representation
# ============================================
# Delete default representation
mol delrep 0 $molid

# Single licorice representation
# This maintains chain connectivity
# Licorice syntax: Licorice <bond_radius> <bond_resolution> <sphere_resolution>
# Setting both bond_resolution and sphere_resolution to 100 for high quality
mol representation Licorice 0.3 100 100

if {$use_binary_coloring} {
    # Color by Type (binary data)
    mol color Type
    puts "Created licorice representation colored by atom Type with customized AOShiny material"
} else {
    # Default color scheme when no binary data
    mol color Index
    puts "Created licorice representation colored by Index with customized AOShiny material"
}

mol selection {all}
mol material AOShiny
mol addrep $molid

# Set nice viewing options
display projection Orthographic
display depthcue off
axes location Off
color Display Background white

# ============================================
# Rendering settings
# ============================================
# Enable shadows
display shadows on

# Enable ambient occlusion
display ambientocclusion on

# Set ambient occlusion quality (higher = better but slower)
display aoambient 0.80
display aodirect 0.30

puts ""
puts "Rendering settings:"
puts "  Shadows: ON"
puts "  Ambient Occlusion: ON"
puts "  Default renderer: Tachyon"

# ============================================
# Helper procedure for 4K rendering
# ============================================
proc render_4k {output_filename} {
    puts "Rendering 4K image to: $output_filename"
    render Tachyon $output_filename.dat
    
    # Run Tachyon with 4K resolution and high quality settings
    set tachyon_cmd "/usr/local/lib/vmd/tachyon_LINUXAMD64"
    if {[file exists $tachyon_cmd]} {
        exec $tachyon_cmd -aasamples 12 -res 3840 2160 -format TARGA -o ${output_filename}.tga $output_filename.dat
        puts "4K rendering complete: ${output_filename}.tga"
    } else {
        puts "Warning: Tachyon executable not found at $tachyon_cmd"
        puts "Please render manually or adjust the tachyon path"
    }
}

puts ""
puts "To render a 4K image, use:"
puts "  render_4k \"output_filename\""
puts ""
puts "Example:"
puts "  render_4k \"structure\""
puts "  (This will create structure.tga at 3840x2160)"

puts "========================================"
if {$use_binary_coloring} {
    puts "Structure loaded and colored successfully!"
    puts "========================================"
    puts "Atom counts:"
    puts "  Type 0 atoms: $count_0 (colored $color_type_0)"
    puts "  Type 1 atoms: $count_1 (colored $color_type_1)"
    puts ""
    puts "To change colors:"
    puts "  Edit lines 12-13 in the script:"
    puts "    set color_type_0 \"<color_name>\""
    puts "    set color_type_1 \"<color_name>\""
    puts ""
    puts "Available colors: blue, red, gray, orange, yellow, tan, silver,"
    puts "  green, white, pink, cyan, purple, lime, mauve, ochre, iceblue,"
    puts "  black, yellow2, yellow3, green2, green3, cyan2, cyan3, blue2,"
    puts "  blue3, violet, violet2, magenta, magenta2, red2, red3, orange2, orange3"
} else {
    puts "Structure loaded successfully!"
    puts "========================================"
    puts "Total atoms: $n_atoms"
    puts "Colored by: Index (default gradient)"
    puts ""
    puts "To color by binary data:"
    puts "  vmd -e vmd_color_binary.tcl -args <xyz_file> <binary_data_file>"
}
puts ""
puts "To change representation style:"
puts "  Graphics -> Representations -> Drawing Method"
puts ""
puts "To save image:"
puts "  File -> Render"
puts "========================================"

