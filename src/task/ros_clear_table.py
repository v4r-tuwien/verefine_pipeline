# run pipeline

# sort by [confidence, verification score] TODO and scale by distance?

# 1) if there is one left that is above threshold...

    # a) grasp

    # b) ask human whether s/he wants it

        # a) no: place in box

        # b) yes: handover routine

# 2) ... only below threshold

    # a) reject -> stop

    # b) ask if we should grasp nonetheless

        # a) no -> stop

        # b) yes -> go to 1)

# 3) ... none left -> stop
