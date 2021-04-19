function SDDP_stage_t_problems(T)
    x = Array{Any,1}(undef,T);
    y = Array{Any,1}(undef,T);
    s = Array{Any,1}(undef,T);
    g = Array{Any,1}(undef,T);
    p = Array{Any,1}(undef,T);
    ϴ = Array{Any,1}(undef,T);
    RHS = Array{Any,1}(undef,T);
    Χ = Array{Any,1}(undef,T);
    Θ = Array{Any,1}(undef,T);

    for t=1:T
        subproblem = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV), "OutputFlag" => 0));
        # Define the variables.
        @variables(subproblem,
                begin
               v̲[h]  <= xᵗ[h=1:H] <= v̅[h]
                   0 <= yᵗ[h=1:H] <= y̅ᵗ[h]
                   0 <= sᵗ[h=1:H] 
                   0 <= gᵗ[f=1:F] <= g̅ᵗ[f]
                   0 <= pᵗ
                   0 <= ϴᵗ
                        RHSᵗ[h=1:H]
                end
              );    
        # Define the constraints.
        FB = Array{Any,1}(undef,H);
        #Hydro Plants with reservoir
        for h in Hᴿ
            if length(U[h])>=1
                FB[h]= @constraint(subproblem, xᵗ[h]-c₀*(sum(yᵗ[m]+sᵗ[m] for m in U[h])-(yᵗ[h]+sᵗ[h]))==RHSᵗ[h]);
            else
                FB[h]= @constraint(subproblem, xᵗ[h]-c₀*(-(yᵗ[h]+sᵗ[h]))==RHSᵗ[h]);
            end
        end

        #Hydro Plants with no reservoir
        for h in Hᴵ
            if length(U[h])>=1
                FB[h]= @constraint(subproblem, -(sum(yᵗ[m]+sᵗ[m] for m in U[h])-(yᵗ[h]+sᵗ[h]))==RHSᵗ[h]);
            else
                FB[h]= @constraint(subproblem, -(-(yᵗ[h]+sᵗ[h]))==RHSᵗ[h]);
            end
        end        
        
        #Demand constraint
        @constraint(subproblem, sum(rʰ[h]*yᵗ[h] for h=1:H)+sum(gᵗ[f] for f=1:F)+pᵗ>=dᵗ);
        
        #objective
        if t < T
            @objective(subproblem,Min,sum(cᶠ[f]*gᵗ[f] for f=1:F)+cᵖ*pᵗ+ϴᵗ);
        else
            @objective(subproblem,Min,sum(cᶠ[f]*gᵗ[f] for f=1:F)+cᵖ*pᵗ);
        end

        x[t] = xᵗ;
        y[t] = yᵗ;
        s[t] = sᵗ;
        g[t] = gᵗ;
        p[t] = pᵗ;
        ϴ[t] = ϴᵗ;
        RHS[t] = RHSᵗ;
        Χ[t] = FB;
        Θ[t] = subproblem; 
    end
    return x, y, s, g, p, ϴ, RHS, Χ, Θ;
end

function SDDP_forward_pass(xval,ϴ̂,Cᵀx₁,T,x,y,s,g,p,ϴ,RHS,Χ,Θ)    
    lb = 1e-10;
    ub = 1e10;

    for t=1:T
        if t > 1
            j = R*(t-2)+1+rand(1:R)
            for h=1:H
                b̃ᵗ = Ξ[j,h];
                if h in Hᴿ
                    fix(RHS[t][h], xval[t-1,h]+c₀*b̃ᵗ)
                else
                    fix(RHS[t][h], b̃ᵗ)
                end
            end
        end
        optimize!(Θ[t]);
        status = termination_status(Θ[t]);
        if termination_status(Θ[t]) != MOI.OPTIMAL
            println(" in Forward Pass")
            println("Model in stage ", t, " in forward pass is ", status)
            exit(0)
        else
            xval[t,:] = value.(x[t]);
            ϴ̂[t] = value(ϴ[t]);
            if t == 1
                lb = objective_value(Θ[t]);
                Cᵀx₁ = sum(cᶠ[f]*value.(g[t])[f] for f=1:F)+cᵖ*value.(p[t]);
            end
        end
    end

    return xval, ϴ̂, Cᵀx₁, lb
end

function SDDP_backward_pass(xval,ϴ̂,T,x,y,s,g,p,ϴ,RHS,Χ,Θ)
    Q = zeros(R);
    Q_prob = fill(1/R,R);
    π = zeros(H,R);
    
    for t=T:-1:2
        for j=1:R
            jj= R*(t-2)+1+j
            for h=1:H
                b̃ᵗ = Ξ[jj,h];
                if h in Hᴿ
                    fix(RHS[t][h], xval[t-1,h]+c₀*b̃ᵗ)
                else
                    fix(RHS[t][h], b̃ᵗ)
                end
            end
            optimize!(Θ[t]);
            status = termination_status(Θ[t])
            if termination_status(Θ[t]) != MOI.OPTIMAL
                println(" in Backward Pass")
                println("Model in stage ", t, " in Backward pass is ", status)
                exit(0)
            end    
            Q[j]=objective_value(Θ[t]);
            for h=1:H
                π[h,j]= dual(Χ[t][h]);
            end
        end    

        Q̌ = sum(Q[j]*Q_prob[j] for j=1:R);
        if (Q̌-ϴ̂[t-1])/max(1e-10,abs(ϴ̂[t-1])) > ϵ 
            @constraint(Θ[t-1],
            ϴ[t-1]-sum(sum(π[h,j]*x[t-1][h] for h in Hᴿ)*Q_prob[j] for j=1:R)
            >=
            Q̌-sum(sum(π[h,j]*xval[t-1,h] for h in Hᴿ)*Q_prob[j] for j=1:R)
            );
        end  
    end
end

