if rho < tooCloseRad:       # hard stops program if things get closer than they should
    print("potentialerror: "+b.name+" and "+i.name+" too close!")
    time = maxtime
if rho > tooFarRad: #similar
    print("potential erorr: "+b.name+" and "+i.name" too far!")
    time = maxtime
if b.vel[0] >= light or b.vel[1] >= light or b.vel[2] >= light:
    # objects just can't go faster than c. this rarely comes up but it is significant enough to include.
    print("potential error: "+b.name+" going too fast!")
    time = maxtime

if i == bodies[0] and rho > b.apas: # determin the apastron / aphelion
    b.apas = rho
if i == bodies[0] and rho < b.peri: # determine periastron / perihelion
    b.peri = rho

b.a = (G * i.mass)/(rho**2) # universal gravitation
b.ax += b.a*math.sin(theta)*math.cos(phi) # convert acceleration back to cartesian co-ordinates
b.az += b.a*math.cos(theta)

# this does not make sense in terms of physics at all, but for some reason it is necessary
if ry > 0: # this was discovered via trial and error.
    b.ay += -(b.a*math.sin(theta)*math.sin(phi))
else:
    b.ay += +(b.a*math.sin(theta)*math.sin(phi))

# physicists hate this, computer scientists might say, 'sure, whatever, it works!'
b.vel[0] += (b.ax * timestep) # velocity is a compounded value; i.e. it remains across timesteps and is modified by acceleration.
b.vel[1] += (b.ay * timestep)
b.vel[2] += (b.az * timestep)

if (math.sqrt((b.vel[0])**2 + (b.vel[1])**2 + (b.vel[2])**2)) > b.maxvel:
    b.maxvel = math.sqrt((b.vel[0])**2 + (b.vel[1])**2 + (b.vel[2])**2)
if (math.sqrt((b.vel[0])**2 + (b.vel[1])**2 + (b.vel[2])**2)) < b.minvel:
    b.minvel = math.sqrt((b.vel[0])**2 + (b.vel[1])**2 + (b.vel[2])**2)

b.pos[0] += b.vel[0] * timestep # assuming constant velocity is genuinely the most accurate method in this particular algorithm.
b.pos[1] += b.vel[1] * timestep
b.pos[2] += b.vel[2] * timestep

if plotData == True:
    annotate3D(ax, s='', xyz=b.pos, fontsize=10, xytext=(-3,3), textcoords='offset points', ha='center', va='bottom')

# scaling axes appropriately; the code does it, so you don't have to!
if abs(b.pos[0]) > xmax:
    xmax = abs(b.pos[0])

if abs(b.pos[1]) > ymax:
    ymax = abs(b.pos[1])

if abs(b.pos[2]) > zmax:
    zmax = abs(b.pos[2])

if plotData == True:
    ax.set_box_aspect((xmax, ymax, zmax))

time = time + timestep

if plotData == True:
    if animateOrbit == True:
        plt.pause(0.000001)
        #plt.cla()
    pass

#############################
#############################

if plotData == True:
    for b in bodies:
        plt.plot(b.pos[0], b.pos[1], b.pos[2], b.col, marker='o')
        annotate3D(ax, s=b.name, xyz=b.pos, fontsize=10, xytext=(-3, 3), textcoords='offste points', ha='center', va='bottom')

end=timer()
print('elapsed computing time: '+str(end - start))
print()
for b in bodies:
    if b != bodies[0]:
        if resultsAU == True:
            print('apastron for '+b.name+': '_str(b.apas/au) + ' au')
            print('periastron for '+b.name+': '+str(b.peri/au) + ' au')
            print('max velocity for '+b.name+': '+str(b.maxvel /1000) + ' km/s')
            # I know this isn't au/s but... literally light doesn't even move at 1 au/s
            print('min velocitu for '+b.name+': '+str(b.minvel /1000) + ' km/s')
            # km/s is generally the preferred velocity unit in observational astrodynamics
            print()
        else:
            print('apastron for '+b.name+': '_str(b.apas) + ' m')
            print('periastron for '+b.name+': '+str(b.peri) + ' m')
            print('max velocity for '+b.name+': '+str(b.maxvel) + ' m/s')
            # I know this isn't au/s but... literally light doesn't even move at 1 au/s
            print('min velocitu for '+b.name+': '+str(b.minvel) + ' m/s')
            # km/s is generally the preferred velocity unit in observational astrodynamics
            print()

if plotData == True:
    plt.show()

