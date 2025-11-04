# VMD Script to color structure by values (with optional dual coloring)
# Usage: 
#   Single color: vmd -e vmd_color_by_values.tcl -args <xyz_file> <color_data_file>
#   Dual color:   vmd -e vmd_color_by_values.tcl -args <xyz_file> <color_data_file1> <color_data_file2>
# Or within VMD: source vmd_color_by_values.tcl
#
# Example: vmd -e vmd_color_by_values.tcl -args structure.xyz color_data1.txt color_data2.txt

# ============================================
# PARSE COMMAND-LINE ARGUMENTS
# ============================================
set dual_mode 0
if {$argc >= 3} {
    set xyz_file [lindex $argv 0]
    set color_data_file [lindex $argv 1]
    set color_data_file2 [lindex $argv 2]
    set dual_mode 1
    puts "Using DUAL COLORING mode with command-line arguments:"
    puts "  XYZ file: $xyz_file"
    puts "  Color data file 1 (Beta): $color_data_file"
    puts "  Color data file 2 (Occupancy): $color_data_file2"
} elseif {$argc >= 2} {
    set xyz_file [lindex $argv 0]
    set color_data_file [lindex $argv 1]
    puts "Using SINGLE COLORING mode with command-line arguments:"
    puts "  XYZ file: $xyz_file"
    puts "  Color data file: $color_data_file"
} elseif {$argc >= 1} {
    puts "ERROR: Not enough arguments provided."
    puts "Usage (single): vmd -e vmd_color_by_values.tcl -args <xyz_file> <color_data_file>"
    puts "Usage (dual):   vmd -e vmd_color_by_values.tcl -args <xyz_file> <color_data_file1> <color_data_file2>"
    exit 1
} else {
    # Default values if no arguments provided
    set xyz_file "../../output/data/mESC/HIPPS/klf1.xyz"
    set color_data_file "../../output/data/mESC/klf1_largest_freq.txt"
    puts "Using default file paths (SINGLE COLORING mode):"
    puts "  XYZ file: $xyz_file"
    puts "  Color data file: $color_data_file"
}

# ============================================
# CONFIGURATION - Modify these as needed
# ============================================
set use_log_scale 0  ;# Set to 1 for log scale, 0 for linear
set colormap "BWR"   ;# Options: "BWR" (blue-white-red), "BlueRed" (blue-to-red, no white), "Rainbow"
set custom_midpoint 0.5   ;# Set to specific value (e.g., 1.0) or -1 to use median
set force_linear_scale 1 ;# If 1, linearly map data to [0,1] and use BWR

# ============================================
# Load structure
# ============================================
mol new $xyz_file type xyz first 0 last -1 step 1 filebonds 0 autobonds 0 waitfor all
set molid [molinfo top]

# ============================================
# Read color data file 1
# ============================================
set fp [open $color_data_file r]
set color_values {}
while {[gets $fp line] >= 0} {
    if {[string index $line 0] != "#"} {
        lappend color_values [string trim $line]
    }
}
close $fp

set n_values [llength $color_values]
puts "Loaded $n_values color values from $color_data_file"

# ============================================
# Read color data file 2 (if in dual mode)
# ============================================
if {$dual_mode} {
    set fp2 [open $color_data_file2 r]
    set color_values2 {}
    while {[gets $fp2 line] >= 0} {
        if {[string index $line 0] != "#"} {
            lappend color_values2 [string trim $line]
        }
    }
    close $fp2
    
    set n_values2 [llength $color_values2]
    puts "Loaded $n_values2 color values from $color_data_file2"
}

# ============================================
# Set up atom selection and assign values to fields
# ============================================
set sel [atomselect $molid "all"]
set n_atoms [$sel num]

if {$n_atoms != $n_values} {
    puts "WARNING: Number of atoms ($n_atoms) does not match number of color values ($n_values)"
}

# Assign values to Beta field (for coloring dataset 1)
$sel set beta $color_values

# If dual mode, assign second dataset to Occupancy field
if {$dual_mode} {
    if {$n_atoms != $n_values2} {
        puts "WARNING: Number of atoms ($n_atoms) does not match number of color values 2 ($n_values2)"
    }
    $sel set occupancy $color_values2
}

# ============================================
# Build explicit chain bonds: 1-2, 2-3, ..., (N-1)-N
# ============================================
if {[catch {package require topotools}]} {
    puts "WARNING: topotools not found; cannot add bonds."
} else {
    topo clearbonds
    for {set i 0} {$i < [expr {$n_atoms - 1}]} {incr i} {
        topo addbond $i [expr {$i + 1}]
    }
    puts "Added chain bonds: [$n_atoms] atoms -> [expr {$n_atoms - 1}] bonds"
}

# ============================================
# Helper function to normalize data
# ============================================
proc normalize_data {sel field_name color_vals use_log force_linear} {
    set min_val [lindex [lsort -real $color_vals] 0]
    set max_val [lindex [lsort -real $color_vals] end]
    
    puts "Dataset ($field_name) value range: \[$min_val, $max_val\]"
    
    if {$use_log} {
        # For log scale, transform the values
        set log_values {}
        foreach val $color_vals {
            if {$val > 0} {
                lappend log_values [expr log10($val)]
            } else {
                lappend log_values [expr log10(1e-10)]
            }
        }
        $sel set $field_name $log_values
        set min_val [expr log10($min_val)]
        set max_val [expr log10($max_val)]
        puts "Using log scale: \[10^$min_val, 10^$max_val\]"
    }
    
    if {$force_linear} {
        # Simple linear normalization to [0,1]
        set current_vals [$sel get $field_name]
        set normalized_vals {}
        set denom [expr $max_val - $min_val]
        if {$denom == 0} {
            foreach val $current_vals { lappend normalized_vals 0.5 }
        } else {
            foreach val $current_vals {
                set norm_val [expr ($val - $min_val) / $denom]
                if {$norm_val < 0.0} { set norm_val 0.0 }
                if {$norm_val > 1.0} { set norm_val 1.0 }
                lappend normalized_vals $norm_val
            }
        }
        $sel set $field_name $normalized_vals
        puts "Applied linear normalization to [0,1] for $field_name"
        return [list 0.0 1.0]
    } else {
        return [list $min_val $max_val]
    }
}

# ============================================
# Normalize dataset 1 (Beta)
# ============================================
set range1 [normalize_data $sel "beta" $color_values $use_log_scale $force_linear_scale]
set min_val1 [lindex $range1 0]
set max_val1 [lindex $range1 1]

# ============================================
# Normalize dataset 2 (Occupancy) if in dual mode
# ============================================
if {$dual_mode} {
    set range2 [normalize_data $sel "occupancy" $color_values2 $use_log_scale $force_linear_scale]
    set min_val2 [lindex $range2 0]
    set max_val2 [lindex $range2 1]
}

# ============================================
# Set up color scale
# ============================================
if {$colormap == "BWR"} {
    color scale method BWR
} elseif {$colormap == "RWB"} {
    color scale method RWB
} elseif {$colormap == "BlueRed"} {
    color scale method BGR
} else {
    color scale method RGB
}

if {$force_linear_scale} {
    color scale min 0.0
    color scale max 1.0
} else {
    # Use range from dataset 1 for global color scale
    color scale min $min_val1
    color scale max $max_val1
}

# ============================================
# Visualization settings
# ============================================
# Delete default representation
mol delrep 0 $molid

if {$dual_mode} {
    # Create TWO representations - one for each dataset
    # Representation 0: Colored by Beta (dataset 1)
    mol representation Licorice 0.2 12.0 12.0
    mol color Beta
    mol selection {all}
    mol material Opaque
    mol addrep $molid
    
    # Representation 1: Colored by Occupancy (dataset 2)
    mol representation Licorice 0.2 12.0 12.0
    mol color Occupancy
    mol selection {all}
    mol material Opaque
    mol addrep $molid
    
    # Initially show only the first dataset (hide the second)
    mol showrep $molid 1 0
    
    puts ""
    puts "Created 2 representations:"
    puts "  Rep 0: Colored by Beta (Dataset 1) - VISIBLE"
    puts "  Rep 1: Colored by Occupancy (Dataset 2) - HIDDEN"
} else {
    # Single representation colored by Beta
    mol representation Licorice 0.2 12.0 12.0
    mol color Beta
    mol selection {all}
    mol material Opaque
    mol addrep $molid
}

# Set nice viewing options
display projection Orthographic
display depthcue off
axes location Off
color Display Background white

# Remove DynamicBonds; explicit bonds are built above

puts ""
puts "========================================"
puts "Structure loaded and colored successfully!"
puts "========================================"
if {$dual_mode} {
    puts "MODE: Dual Coloring"
    puts "  Dataset 1 (Rep 0, Beta field): $color_data_file"
    puts "  Dataset 2 (Rep 1, Occupancy field): $color_data_file2"
    puts ""
    puts "HOW TO SWITCH between datasets in VMD GUI:"
    puts "  1. Open: Graphics -> Representations"
    puts "  2. You will see two representations:"
    puts "     - Rep 0: 'Licorice' with 'Coloring Method: Beta'"
    puts "     - Rep 1: 'Licorice' with 'Coloring Method: Occupancy'"
    puts "  3. Click the eye icon to toggle visibility:"
    puts "     - Eye icon visible = representation is shown"
    puts "     - Eye icon hidden = representation is hidden"
    puts "  4. You can show one at a time or both simultaneously"
    puts ""
    puts "TIP: Click on a representation name to see/modify its settings"
} else {
    puts "MODE: Single Coloring"
    puts "  Coloring by: Beta values"
    puts "  Data file: $color_data_file"
}
puts ""
puts "Settings:"
puts "  Colormap: $colormap"
if {$use_log_scale} {
    puts "  Scale: Logarithmic"
} else {
    puts "  Scale: Linear"
}
if {$force_linear_scale} {
    puts "  Normalization: Linear [0,1]"
}
puts ""
puts "To save image:"
puts "  File -> Render"
puts "========================================"

